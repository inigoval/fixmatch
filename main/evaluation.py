import numpy as np
import torch
import wandb
import pytorch_lightning as pl
from scipy.linalg import sqrtm

from config import load_config

config = load_config()

## Classification metrics ##


def accuracy(y_hat, y):
    _, y_pred = torch.max(y_hat, 1)
    n_test = y.size(0)
    n_correct = (y_pred == y).sum()
    accuracy = n_correct / n_test
    return accuracy


################################
#### Class balance metrics #####
################################


def count_elements(tens, value):
    bool_tens = tens == value
    n = torch.nonzero(bool_tens).numel()
    return n


def pred_label_fraction(y_hat, label):
    _, y_pred = torch.max(y_hat, 1)
    n_label = count_elements(y_pred, label)
    # y_pred = y_pred.detach().cpu().numpy()
    # n_label = np.count_nonzero(y_pred == label)
    return n_label / torch.numel(y_pred)


################################
## Frechet Inception Distance ##
################################


def compute_mu_sig(f_layer):
    mu = np.mean(f_layer, axis=0)
    sigma = np.cov(f_layer, rowvar=False)
    return mu, sigma


def calculate_fid(f_real, f_fake):
    f_real = f_real.detach().cpu().numpy()
    f_fake = f_fake.detach().cpu().numpy()
    mu_real, sig_real = compute_mu_sig(f_real)
    mu_fake, sig_fake = compute_mu_sig(f_fake)

    S = sqrtm((np.dot(sig_fake, sig_real)))

    if np.iscomplexobj(S):
        S = S.real

    Dmu = np.square(mu_fake - mu_real).sum()

    fid = Dmu + np.trace((sig_fake + sig_real - 2 * S), axis1=0, axis2=1)
    return fid
