import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch.utils.data as D

from config import load_config
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
    mb_cut,
    subset,
    data_splitter_strat,
    unbalance,
)
from fixmatch import TransformFixMatch
from utilities import compute_mu_sig

paths = Path_Handler()
path_dict = paths._dict()
# config = load_config()


# Define transforms
totens = T.ToTensor()
weak_transform = lambda mu, sig: T.Compose(
    [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(180),
        T.ToTensor(),
        Circle_Crop(),
        T.Normalize((mu,), (sig,)),
    ]
)
test_transform = lambda mu, sig: T.Compose(
    [T.ToTensor(), Circle_Crop(), T.Normalize((mu,), (sig,))]
)

transforms = lambda mu, sig: {
    "weak": T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(180),
            T.ToTensor(),
            Circle_Crop(),
            T.Normalize((mu,), (sig,)),
        ]
    ),
    "u": TransformFixMatch(mu, sig),
    "test": T.Compose([T.ToTensor(), Circle_Crop(), T.Normalize((mu,), (sig,))]),
}


class mb_rgzDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.path = path_dict["data"]
        self.config = config
        self.transforms = {}
        self.hparams = {}

    def prepare_data(self):
        MB_nohybrids(self.path, train=True, download=True)
        MB_nohybrids(self.path, train=False, download=True)
        RGZ20k(self.path, train=True, download=True)

    def setup(self, stage=None):
        # Compute mean and standard deviation of combined datasets
        mu, sig = compute_mu_sig(
            D.ConcatDataset(
                [
                    MB_nohybrids(self.path, train=True, transform=totens),
                    RGZ20k(self.path, train=True, transform=totens),
                ]
            )
        )

        # Initialise transforms #
        self.transforms = transforms(mu, sig)

        # Initialise different subsets
        mb = {
            "confident": lambda transform: MBFRConfident(
                self.path, train=True, transform=transform
            ),
            "unconfident": lambda transform: MBFRUncertain(
                self.path, train=True, transform=transform
            ),
            "all": lambda transform: MB_nohybrids(
                self.path, train=True, transform=transform
            ),
        }

        self.data, self.data_idx = data_splitter_strat(
            mb[self.config["data"]["l"]](self.transforms["weak"]),
            split=self.config["data"]["split"],
            val_frac=self.config["data"]["val_frac"],
        )

        self.data["test"] = MB_nohybrids(
            path_dict["mb"], train=False, transform=self.transforms["test"]
        )

        # Cut unlabelled RGZ samples and define unlabelled training set #
        rgz = RGZ20k(path_dict["rgz"], train=True, transform=self.transforms["u"])
        self.size_cut(rgz)
        mb_cut(rgz)

        # Adjust unlabelled data set size to match mu value
        len_u = torch.clamp(
            torch.tensor(self.config["mu"] * len(self.data["l"])),
            min=0,
            max=len(self.data["u"]),
        ).item()
        print(len_u)
        rgz = subset(rgz, len_u)

        # Reassign unlabelled dataset with rgz data
        self.data["u"] = rgz

        # Flip a number of targets randomly #
        if self.config["train"]["flip"]:
            self.data["l"] = flip_targets(self.data["l"], self.config["train"]["flip"])

        self.save_hparams()

    def train_dataloader(self):
        l_batch_size = self.config["train"]["batch_size"]
        u_batch_size = int(self.config["mu"] * l_batch_size)
        loader_l = DataLoader(self.data["l"], l_batch_size, shuffle=True)
        loader_u = DataLoader(self.data["u"], u_batch_size, shuffle=True)
        loaders = {"u": loader_u, "l": loader_l}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders

    def val_dataloader(self):
        loader_val = DataLoader(self.data["val"], int(len(self.data["val"])))
        # loader_u = DataLoader(self.data["u"], int(len(self.data["u"])))
        loaders = {"val": loader_val}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders

    def test_dataloader(self):
        loader = DataLoader(self.data["test"], int(len(self.data["test"])))
        return loader

    def save_hparams(self):
        self.hparams.update(
            {
                "n_labelled": len(self.data["l"]),
                "n_unlabelled": len(self.data["u"]),
                "n_test": len(self.data["test"]),
                "f_u": label_fraction(self.data["u"], 0),
                "f_l": label_fraction(self.data["l"], 0),
            }
        )

    def size_cut(self, dset):
        length = len(dset)
        idx = np.argwhere(dset.sizes > self.config["cut_threshold"]).flatten()
        dset.data = dset.data[idx, ...]
        dset.names = dset.names[idx, ...]
        dset.rgzid = dset.rgzid[idx, ...]
        dset.sizes = dset.sizes[idx, ...]
        dset.mbflg = dset.mbflg[idx, ...]
        self.mu = idx.shape[0] / length


class mbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        path=path_dict["data"],
    ):
        super().__init__()
        self.path = path
        self.config = config
        self.hparams = {}
        self.transforms = {}

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
            "unconfident": lambda transform: MBFRUncertain(
                self.path, train=True, transform=transform
            ),
            "all": lambda transform: MB_nohybrids(
                self.path, train=True, transform=transform
            ),
            "rgz": lambda transform: RGZ20k(self.path, train=True, transform=transform),
        }

        self.data, self.data_idx = data_splitter_strat(
            mb[self.config["data"]["l"]](totens),
            split=self.config["data"]["split"],
            val_frac=self.config["data"]["val_frac"],
        )

        # Draw unlabelled samples from different set if required
        if self.config["data"]["l"] != self.config["data"]["l"]:
            mb_u = mb[self.config["data"]["u"]]

            # Adjust unlabelled data set size to match mu value
            len_u = torch.clamp(
                torch.tensor(self.config["mu"] * len(self.data["l"])),
                min=0,
                max=len(self.data["u"]),
            ).item()

            self.data["u"] = subset(mb_u, len_u)
            self.data_idx["u"] = self.data["u"].indices

        # Otherwise simply change transform to unlabelled transform for chosen unlabelled points
        else:
            self.data["u"] = D.Subset(
                mb[self.config["data"]["l"]](totens), self.data_idx["u"]
            )

        ## Unbalance the unlabelled dataset and change mu accordingly ##
        if self.config["data"]["fri_R"] >= 0:
            self.data_idx["u"] = unbalance(
                self.data["u"],
                self.config["data"]["fri_R"],
            )
            self.config["mu"] = len(self.data_idx["u"]) / len(self.data_idx["l"])
            print(f"mu = {len(self.data_idx['u'])/len(self.data_idx['l'])}")

        # Load dataset and calculate mean and std & initialise transforms #
        mu, sig = compute_mu_sig(
            D.ConcatDataset([self.data["l"], self.data["u"], self.data["val"]])
        )
        self.transforms = transforms(mu, sig)

        self.data["u"] = mb[self.config["data"]["u"]](self.transforms["u"])
        self.data["l"] = mb[self.config["data"]["l"]](self.transforms["weak"])
        self.data["test"] = mb["all"](self.transforms["test"])

        # Flip a number of targets randomly
        if self.config["train"]["flip"]:
            self.data["l"] = flip_targets(self.data["l"], self.config["train"]["flip"])

        # Compute & save data hyperparameters and #
        self.save_hparams()

    def train_dataloader(self):
        l_batch_size = self.config["train"]["batch_size"]
        u_batch_size = int(self.config["mu"] * l_batch_size)
        loader_l = DataLoader(self.data["l"], l_batch_size, shuffle=True)
        loader_u = DataLoader(self.data["u"], u_batch_size, shuffle=True)
        loaders = {"u": loader_u, "l": loader_l}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
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
        self.hparams.update(
            {
                "n_labelled": len(self.data["l"]),
                "n_unlabelled": len(self.data["u"]),
                "n_test": len(self.data["test"]),
                "f_u": label_fraction(self.data["u"], 0),
                "f_l": label_fraction(self.data["l"], 0),
            }
        )
