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


#################
### Callbacks ###
#################


class ImageLogger(pl.Callback):
    def __init__(self, grid_size=3):
        super().__init__()
        self.n_grid = grid_size

    def on_validation_epoch_end(self, trainer, pl_module):
        x = pl_module.generate(self.n_grid ** 2).detach().cpu().numpy()
        img = self.plot(x)
        trainer.logger.experiment.log(
            {
                "generated images": [wandb.Image(img)],
            },
        )

    def plot(self, img_array):
        img_list = list(img_array)
        fig = plt.figure(figsize=(13.0, 13.0))
        grid = ImageGrid(fig, 111, nrows_ncols=(self.n_grid, self.n_grid), axes_pad=0)

        for ax, im in zip(grid, img_list):
            im = im.reshape((150, 150))
            ax.axis("off")
            ax.imshow(im, cmap="hot")
        plt.axis("off")
        pil_img = fig2img(fig)
        plt.close(fig)
        return pil_img


class MetricLogger(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        x = pl_module.generate(1000)
        y_hat = pl_module.forward(x)

        # Log class balance of predictions #
        f_fake = pred_label_fraction(y_hat, 0)
        pl_module.log("generator/f_fake", f_fake, on_step=False, on_epoch=True)

        # Softmax entropy #
        H = entropy(y_hat)
        H = H.tolist()
        data = [[h] for h in H]
        table = wandb.Table(data=data, columns=["entropy"])
        pl_module.logger.experiment.log(
            {
                "generator/entropy": wandb.plot.histogram(
                    table, "entropy", title="softmax entropy (fake)"
                )
            }
        )
        pl_module.logger.experiment.log({"softmax entropy": wandb.Histogram(H)})

        if config["train"]["fid"]:
            fid_l = pl_module.fid(pl_module.trainer.datamodule.train_l)
            fid_u = pl_module.fid(pl_module.trainer.datamodule.train_u)
            pl_module.log("u/fid", fid_u)
            pl_module.log("l/fid", fid_l)


class FeaturePlot(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        x_real, _ = dset2tens(pl_module.trainer.datamodule.train_l)
        x_real = x_real.type_as(pl_module.G.up1[0].weight)
        x_fake = pl_module.generate(x_real.shape[0])
        logits_real, y_real, f_real = pl_module.D(x_real)
        logits_fake, y_fake, f_fake = pl_module.D(x_fake)

        table_real = wandb.Table(data=logits_real.tolist(), columns=["fr1", "fr2"])
        table_fake = wandb.Table(data=logits_fake.tolist(), columns=["fr1", "fr2"])

        pl_module.logger.experiment.log(
            {
                "real": wandb.plot.scatter(table_real, "fr1", "fr2"),
                "fake": wandb.plot.scatter(table_fake, "fr1", "fr2"),
            }
        )
