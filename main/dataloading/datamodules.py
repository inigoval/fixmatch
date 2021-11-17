import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn.model_selection import train_test_split
import torch.utils.data as D

from paths import Path_Handler
from dataloading.datasets import (
    MB_nohybrids,
    RGZ20k,
    MiraBest_full,
    MBFRUncertain,
    MBFRConfident,
)
from dataloading.utils import Circle_Crop, label_fraction, flip_targets
from dataloading.utils import (
    size_cut,
    random_subset,
    data_splitter_strat,
    unbalance_idx,
    uval_splitter_strat,
)
from fixmatch import TransformFixMatch
from utilities import compute_mu_sig

paths = Path_Handler()
path_dict = paths._dict()


# Define transforms
totens = T.ToTensor()


class mbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        path=path_dict["data"],
    ):
        super().__init__()
        self.path = path
        self.config = config
        self.hyperparams = {}

    def prepare_data(self):
        MB_nohybrids(self.path, train=False, download=True)
        MB_nohybrids(self.path, train=True, download=True)
        MBFRUncertain(self.path, train=True, download=True)
        MBFRConfident(self.path, train=True, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self, stage=None):

        mb = {
            "confident": lambda transform: MBFRConfident(
                self.path, train=True, transform=transform
            ),
            "uncertain": lambda transform: MBFRUncertain(
                self.path, train=True, transform=transform
            ),
            "all": lambda transform: MB_nohybrids(
                self.path, train=True, transform=transform
            ),
            "test": lambda transform: MB_nohybrids(
                self.path, train=False, transform=transform
            ),
            "rgz": lambda transform: RGZ20k(self.path, train=True, transform=transform),
        }

        datasets = {
            "l": mb[self.config["data"]["l"]],
            "u": mb[self.config["data"]["u"]],
        }

        self.data, self.data_idx = data_splitter_strat(
            datasets["l"](totens),
            split=self.config["data"]["split"],
            val_frac=self.config["data"]["val_frac"],
            seed=self.config["seed"],
        )

        # Draw unlabelled samples from different set if required
        if self.config["data"]["l"] != self.config["data"]["u"]:
            n_max = len(datasets["u"](totens))
            self.data_idx["u"] = np.arange(n_max)

            #            # Concat uncertain and confident to avoid data leak
            #            if self.config["data"]["l"] == "confident":
            #                uval_splitter_strat(
            #                    D.ConcatDataset([self.data["u"], mb["uncertain"](totens)]),
            #                    self.data,
            #                    self.data_idx,
            #                    seed=self.config["seed"],
            #                    val_frac=self.config["data"]["val_frac"],
            #                )
            #
            #            if self.config["data"]["l"] == "uncertain":
            #                self.data["u"] = D.ConcatDataset(
            #                    [self.data["u"], mb["confident"](totens)]
            #                )
            #
            #                uval_splitter_strat(
            #                    D.ConcatDataset([self.data["u"], mb["uncertain"](totens)]),
            #                    self.data,
            #                    self.data_idx,
            #                    seed=self.config["seed"],
            #                    val_frac=self.config["data"]["val_frac"],
            #                )
            #

            # Apply angular size lower limit if using rgz
            if self.config["data"]["u"] == "rgz":
                self.data_idx["u"] = size_cut(
                    self.config["cut_threshold"], datasets["u"](totens)
                )

            # Adjust unlabelled data set size to match mu value (probably don't need this anymore)
            if self.config["data"]["clamp_u"]:
                n = torch.clamp(
                    torch.tensor(
                        self.config["data"]["clamp_u"]
                        * len(self.data["l"])
                        * self.config["mu"]
                    ),
                    min=0,
                    max=n_max,
                ).item()

                self.data_idx["u"] = np.random.choice(self.data_idx["u"], int(n))

            self.data["u"] = D.Subset(datasets["u"](totens), self.data_idx["u"])

        ## Unbalance the unlabelled dataset ##
        if self.config["data"]["fri_R"] >= 0:
            self.data_idx["u"] = unbalance_idx(
                self.data["u"],
                self.config["data"]["fri_R"],
            )
            self.data["u"] = D.Subset(datasets["u"](totens), self.data_idx["u"])

            # self.config["mu"] = len(self.data_idx["u"]) / len(self.data_idx["l"])
            # print(f"mu = {len(self.data_idx['u'])/len(self.data_idx['l'])}")

        # Load dataset, calculate mean and std & initialise transforms #
        mu, sig = compute_mu_sig(
            D.ConcatDataset([self.data["l"], self.data["u"], self.data["val"]])
        )
        self.mu, self.sig = mu, sig

        self.transforms = {
            "weak": T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(180),
                    T.ToTensor(),
                    Circle_Crop(),
                    T.Normalize((mu,), (sig,)),
                ]
            ),
            "u": TransformFixMatch(self.config, mu, sig),
            "test": T.Compose(
                [
                    T.ToTensor(),
                    Circle_Crop(),
                    T.Normalize((mu,), (sig,)),
                ]
            ),
            "val": T.Compose(
                [
                    T.ToTensor(),
                    Circle_Crop(),
                    T.Normalize((mu,), (sig,)),
                ]
            ),
        }

        self.data["u"] = D.Subset(
            datasets["u"](self.transforms["u"]), self.data_idx["u"]
        )

        self.data["l"] = D.Subset(
            datasets["l"](self.transforms["weak"]), self.data_idx["l"]
        )

        self.data["val"] = D.Subset(
            datasets["l"](self.transforms["val"]), self.data_idx["val"]
        )

        self.data["test"] = mb["test"](self.transforms["test"])

        # Flip a number of targets randomly
        if self.config["train"]["flip"]:
            self.data["l"] = flip_targets(self.data["l"], self.config["train"]["flip"])

        # Compute & save data hyperparameters and #
        self.save_hparams()

        print(self.config["data"]["l"])
        print(self.data_idx["l"])

    def train_dataloader(self):
        l_batch_size = self.config["batch_size"]
        u_batch_size = int(self.config["mu"] * l_batch_size)
        loader_l = DataLoader(self.data["l"], l_batch_size, shuffle=True)
        loader_u = DataLoader(self.data["u"], u_batch_size, shuffle=True)
        loaders = {"u": loader_u, "l": loader_l}
        combined_loaders = CombinedLoader(loaders, "min_size")
        return combined_loaders

    def val_dataloader(self):
        val_batch_size = 100
        loader_val = DataLoader(self.data["val"], val_batch_size)
        return loader_val

    def test_dataloader(self):
        u_batch_size = 200
        loader_test = DataLoader(self.data["test"], int(len(self.data["test"])))
        loader_u = DataLoader(self.data["u"], u_batch_size)
        return [loader_test, loader_u]

    def save_hparams(self):
        self.hyperparams.update(
            {
                "n_labelled": len(self.data["l"]),
                "n_unlabelled": len(self.data["u"]),
                "n_test": len(self.data["test"]),
                "n_val": len(self.data["val"]),
                "f_u": label_fraction(self.data["u"], 0),
                "f_l": label_fraction(self.data["l"], 0),
            }
        )
