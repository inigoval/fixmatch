import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import wandb
import torchmetrics.functional as tmF
import numpy as np

from torch.utils.data import DataLoader
from statistics import mean

from networks.models import disc, Tang
from utilities import logit_loss, entropy, dset2tens, flip
from evaluation import accuracy, pred_label_fraction, calculate_fid
from config import load_config
from randaugmentmc import RandAugmentMC
from dataloading.utils import Circle_Crop
from paths import Path_Handler


class TransformFixMatch(object):
    def __init__(self, mu, sig):
        self.weak = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(180),
                T.ToTensor(),
                Circle_Crop(),
            ]
        )

        self.strong = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(180),
                # T.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
                RandAugmentMC(n=2, m=10),
                T.ToTensor(),
            ]
        )

        self.normalize = T.Normalize((mu,), (sig,))

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class clf(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.best_acc = 0
        self.config = config
        paths = Path_Handler()
        paths.fill_dict()
        paths.dict["chk"] = (
            paths.dict["files"]
            / f"{config['type']}_split{config['data']['split']}_l{config['data']['l']}{config['data']['u']}_ufrac{config['mu']}"
        )
        paths.create_paths()
        self.paths = paths.dict

        if config["model"]["architecture"] == "basic":
            self.C = disc()
        elif config["model"]["architecture"] == "tang":
            self.C = Tang()

    def forward(self, x, logit=False):
        # D(img) returns (logits, y, features)
        if logit:
            return self.C(x)[0]
        else:
            return self.C(x)[1]

    def training_step(self, batch, batch_idx):
        x_l, y_l = batch["l"]
        x_u, _ = batch["u"]
        x_u_w, x_u_s = x_u

        ## Pass through classifier ##
        l_l, _ = self.C(x_l)
        l_u_w, p_u_w = self.C(x_u_w)
        l_u_s, p_u_s = self.C(x_u_s)

        ## Supervised Loss l_l
        ce_loss = F.cross_entropy(l_l, y_l)
        self.log("train/cross entropy loss", ce_loss)

        ## pseudo label loss Loss ##
        p_pseudo_label, pseudo_label = torch.max(p_u_w.detach(), dim=-1)
        threshold_mask = p_pseudo_label.ge(self.config["tau"]).float()
        pseudo_loss = (
            F.cross_entropy(l_u_s, pseudo_label, reduction="none") * threshold_mask
        ).mean()
        self.log("train/pseudo-label loss", pseudo_loss)

        ## Total Loss ##
        loss = ce_loss + self.config["lambda"] * pseudo_loss
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # Loop through unlabelled and test loaders to calculate metrics #
        x, y = batch
        logits, y_pred = self.C(x)

        ## Loss ##
        loss = self.ce_loss(logits, y)
        self.log("val/loss", loss)

        ## F1 score ##
        # f1 = tmF.f1(y_pred, y, num_classes=2, average="none")
        # self.log("val/f1", f1)

        ## Accuracy ##
        acc = tmF.accuracy(y_pred, y)
        self.log("val/accuracy", acc)

        return acc

    def validation_epoch_end(self, outs):
        ## Compute mean over epoch ##
        outs = torch.FloatTensor(outs)
        acc = torch.mean(outs)
        if acc > self.best_acc:
            self.C_weights = self.C.state_dict()
            self.best_acc = acc
        self.log("val/best_accuracy", self.best_acc)

    def test_step(self, batch, batch_idx, dataloader_idx):
        save_path = self.paths["chk"] / f"seed_{self.config['seed']}.pt"
        torch.save(self.C_weights, save_path)
        self.C.load_state_dict(self.C_weights)

        if dataloader_idx == 0:
            x, y = batch
            logits, y_pred = self.C(x)

            loss = F.cross_entropy(logits, y)
            self.log(f"test/loss", loss)

            ## Accuracy ##
            acc = tmF.accuracy(y_pred, y)
            self.log(f"test/accuracy", acc)

            f1 = tmF.f1(y_pred, y, num_classes=2, average="none")
            precision = tmF.precision(y_pred, y, num_classes=2, average="none")
            recall = tmF.recall(y_pred, y, num_classes=2, average="none")
            names = ["fri", "frii"]

            ## F1, precision, recall ##
            for p, r, f, name in zip(precision, recall, f1, names):
                self.log(f"test/{name}_precision", p)
                self.log(f"test/{name}_recall", r)
                self.log(f"test/{name}_f1", f)

        if dataloader_idx == 1:
            x, y = batch
            x = x[0]
            logits, y_pred = self.C(x)

            ## Accuracy and Loss ##
            loss = F.cross_entropy(logits, y)
            acc = tmF.accuracy(y_pred, y)
            self.log(f"unlabelled/loss", loss)
            self.log(f"unlabelled/accuracy", acc)

            ## F1, precision, recall ##
            f1 = tmF.f1(y_pred, y, num_classes=2, average="none")
            precision = tmF.precision(y_pred, y, num_classes=2, average="none")
            recall = tmF.recall(y_pred, y, num_classes=2, average="none")
            names = ["fri", "frii"]

            ## F1, precision, recall ##
            for p, r, f, name in zip(precision, recall, f1, names):
                self.log(f"unlabelled/{name}_precision", p)
                self.log(f"unlabelled/{name}_recall", r)
                self.log(f"unlabelled/{name}_f1", f)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config["train"]["lr"])
        return opt


"""    def test_epoch_end(self, outputs):
        loss = np.mean(np.asarray([l for l, a, f1, p, r in outputs]))
        acc = np.mean(np.asarray([a for l, a, f1, p, r in outputs]))
        self.log("unlabelled/loss", loss)
        self.log("unlabelled/accuracy", acc)

        ## F1, precision, recall ##
        names = ["fri", "frii"]
        f1 = np.mean(np.asarray([f1 for l, a, f1, p, r in outputs]), 0)
        precision = np.mean(np.asarray([p for l, a, f1, p, r in outputs]), 0)
        recall = np.mean(np.asarray([r for l, a, f1, p, r in outputs]), 0)

        for p, r, f, name in zip(precision, recall, f1, names):
            self.log(f"unlabelled/{name}_precision", p)
            self.log(f"unlabelled/{name}_recall", r)
            self.log(f"unlabelled/{name}_f1", f)
"""
