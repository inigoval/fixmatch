import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch import logsumexp
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb

from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader

from paths import Path_Handler
from evaluation import pred_label_fraction, calculate_fid
from config import load_config

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


config = load_config()


def logit_loss(l_real, l_fake):
    loss_real = F.softplus(logsumexp(l_real, 1)) - logsumexp(l_real, 1)
    loss_fake = F.softplus(logsumexp(l_fake, 1))
    assert loss_fake.shape[0] == l_real.shape[0]
    loss = torch.mean(0.5 * (loss_real + loss_fake))
    return loss


def entropy(p, eps=0.0000001, loss=False):
    H_i = -torch.log(p + eps) * p
    H = torch.sum(H_i, 1).view(-1)

    if not loss:
        # Clamp to avoid negative values due to eps
        H = torch.clamp(H, min=0)
        return H.detach().cpu().numpy()

    H = torch.mean(H)

    return H


def intensity_loss(img):
    I_mean = torch.mean(img)
    return torch.mean(img)


def pull_away_loss(f_fake):
    n = f_fake.shape[0]
    f_norm = torch.linalg.norm(f_fake, ord=2, dim=1)
    f = f_fake / f_norm.view(-1, 1).expand_as(f_fake)
    cosine = torch.mm(f, f.t())
    mask = torch.ones(cosine.shape).type_as(f) - torch.diag(torch.ones((n,)).type_as(f))
    loss = torch.sum((cosine * mask) ** 2) / (n * (n - 1))
    return loss


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


def dset2tens(dset):
    """
    Returns a tuple (x, y) containing the entire input dataset
    """
    return next(iter(DataLoader(dset, int(len(dset)))))


def flip(labels, p_flip):
    n_labels = labels.shape[0]
    n_flip = int(p_flip * n_labels)
    if n_flip:
        idx = torch.randint(labels.shape[0], (n_flip))
    else:
        return labels


def compute_mu_sig(dset):
    x, _ = dset2tens(dset)
    return torch.mean(x), torch.std(x)


def batch_eval(fn_dict, dset, batch_size=200):
    """
    Take functions which acts on data x,y and evaluates over the whole dataset in batches, returning a list of results for each calculated metric
    """
    n = len(dset)
    loader = DataLoader(dset, batch_size)

    # Fill the output dictionary with empty lists
    outs = {}
    for key in fn_dict.keys():
        outs[key] = []

    for x, y in loader:
        # Take only weakly augmented sample if passing through unlabelled data
        #        if strong_T:
        #            x = x[0]

        # Append result from each batch to list in outputs dictionary
        for key, fn in fn_dict.items():
            outs[key].append(fn(x, y))

    return outs
