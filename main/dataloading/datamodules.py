import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
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
from galaxy_mnist import GalaxyMNIST

from dataloading.utils import Circle_Crop, label_fraction, flip_targets
from dataloading.utils import size_cut, data_splitter_strat, unbalance_idx

from fixmatch import TransformFixMatch
from utilities import compute_mu_sig

paths = Path_Handler()
path_dict = paths._dict()


# Define transforms
to_tensor = T.ToTensor()

class GalaxyMNISTDataModule(pl.LightningDataModule):

    def __init__(
        self, config, path=path_dict['data']
    ):
        super().__init__()
        self.path = path
        self.config = config
        self.hyperparams = {}

    def prepare_data(self):
        # trigger download now
        _ = GalaxyMNIST(
            root=self.path,
            download=True
        )

    def setup(self, stage=None):
        # Split the data while preserving class balance
        dataset = GalaxyMNIST(root=self.path, train=True)  # only 70% for now...TODO
        self.data, self.data_idx = data_splitter_strat(
            dataset,
            split=self.config["data"]["split"],
            val_frac=self.config["data"]["val_frac"],
            seed=self.config["seed"],
        )
        # print(self.data)

        # Calculate mean and std of data
        mu, sig = compute_mu_sig(
            D.ConcatDataset([self.data["labelled"], self.data["unlabelled"], self.data["val"]])
        )
        self.mu, self.sig = mu, sig
    
        # Initialise transforms with mean and std from data
        self.transforms = default_transforms(self.config, mu, sig)        

        # ## Finally subset all data using correct transform ##
        # self.data["unlabelled"] = D.Subset(
        #     datasets["unlabelled"](self.transforms["unlabelled"]), self.data_idx["unlabelled"]
        # )
        # self.data["labelled"] = D.Subset(
        #     datasets["labelled"](self.transforms["weak"]), self.data_idx["labelled"]
        # )
        # self.data["val"] = D.Subset(
        #     datasets["labelled"](self.transforms["val"]), self.data_idx["val"]
        # )
        # self.data["test"] = mb["test"](self.transforms["test"])
    
    def train_dataloader(self):
        """Batch unlabelled and labelled data together"""

        l_batch_size = self.config["batch_size"]

        # Calculate larger batch size for unlabelled data
        u_batch_size = int(self.config["mu"] * l_batch_size)

        # Define loaders with different batch sizes
        loader_labelled = DataLoader(self.data["labelled"], l_batch_size, shuffle=True)
        loader_unlabelled = DataLoader(self.data["unlabelled"], u_batch_size, shuffle=True)

        # "Zip" loaders together for simultaneous loading
        loaders = {"unlabelled": loader_unlabelled, "labelled": loader_labelled}
        # dict of dataloaders
        combined_loaders = CombinedLoader(loaders, "min_size")  
        # see also max_size_cycle. 
        # Should each epoch be the size of the number of batches in smallest dataloader or the largest dataloader. 
        # Min size easier to compare epochs as cutting the unlabelled data by dif amounts, making metrics tricky to compare.
        return combined_loaders

    def val_dataloader(self):
        val_batch_size = 100
        loader_val = DataLoader(self.data["val"], val_batch_size)
        return loader_val

    def test_dataloader(self):
        """Batch test and unlabelled data sequentially"""
        u_batch_size = 200
        # test step where idx is an arg, list-type. Runs sequentially (all of one, than all of the next), while CombinedLoader does 
        loader_test = DataLoader(self.data["test"], int(len(self.data["test"])))
        loader_unlabelled = DataLoader(self.data["unlabelled"], u_batch_size)
        return [loader_test, loader_unlabelled]

    def save_hparams(self):
        self.hyperparams.update(
            {
                "n_labelled": len(self.data["labelled"]),
                "n_unlabelled": len(self.data["unlabelled"]),
                "n_test": len(self.data["test"]),
                "n_val": len(self.data["val"]),
                "f_u": label_fraction(self.data["unlabelled"], 0),
                "f_l": label_fraction(self.data["labelled"], 0),
            }
        )


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

        # Define dictionary with different subsets of data
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

        # Use config to extract correct subsets for (l)abelled/(u)nlabelled data
        datasets = {
            "labelled": mb[self.config["data"]["labelled"]],
            "unlabelled": mb[self.config["data"]["unlabelled"]],
        }

        # Split the data while preserving class balance
        self.data, self.data_idx = data_splitter_strat(
            datasets["labelled"](to_tensor),
            split=self.config["data"]["split"],
            val_frac=self.config["data"]["val_frac"],
            seed=self.config["seed"],
        )

        # Draw unlabelled samples from different set if required
        if self.config["data"]["labelled"] != self.config["data"]["unlabelled"]:
            n_max = len(datasets["unlabelled"](to_tensor))
            self.data_idx["unlabelled"] = np.arange(n_max)

            # Apply angular size lower limit if using rgz
            if self.config["data"]["unlabelled"] == "rgz":
                self.data_idx["unlabelled"] = size_cut(
                    self.config["cut_threshold"], datasets["unlabelled"](to_tensor)
                )

            # Adjust unlabelled data set size to match mu value (probably don't need this anymore)
            if self.config["data"]["clamp_unlabelled"]:
                n = torch.clamp(
                    torch.tensor(
                        self.config["data"]["clamp_unlabelled"]
                        * len(self.data["labelled"])
                        * self.config["mu"]
                    ),
                    min=0,
                    max=n_max,
                ).item()
                self.data_idx["unlabelled"] = np.random.choice(self.data_idx["unlabelled"], int(n))

            # Re-subset unlabelled data using new indices
            self.data["unlabelled"] = D.Subset(datasets["unlabelled"](to_tensor), self.data_idx["unlabelled"])

        # Unbalance the unlabelled dataset
        if self.config["data"]["fri_R"] >= 0:
            self.data_idx["unlabelled"] = unbalance_idx(
                self.data["unlabelled"],
                self.config["data"]["fri_R"],
            )

            # Re-subset unlabelled data using new indices
            self.data["unlabelled"] = D.Subset(datasets["unlabelled"](to_tensor), self.data_idx["unlabelled"])

        # Calculate mean and std of data
        mu, sig = compute_mu_sig(
            D.ConcatDataset([self.data["labelled"], self.data["unlabelled"], self.data["val"]])
        )
        self.mu, self.sig = mu, sig

        # Initialise transforms with mean and std from data
        self.transforms = default_transforms(self.config, mu, sig)

        ## Finally subset all data using correct transform ##
        self.data["unlabelled"] = D.Subset(
            datasets["unlabelled"](self.transforms["unlabelled"]), self.data_idx["unlabelled"]
        )

        self.data["labelled"] = D.Subset(
            datasets["labelled"](self.transforms["weak"]), self.data_idx["labelled"]
        )

        self.data["val"] = D.Subset(
            datasets["labelled"](self.transforms["val"]), self.data_idx["val"]
        )

        self.data["test"] = mb["test"](self.transforms["test"])

        # Flip a number of targets randomly
        if self.config["train"]["flip"]:
            self.data["labelled"] = flip_targets(self.data["labelled"], self.config["train"]["flip"])

        # Compute & save data hyperparameters and #
        self.save_hparams()

        # Print indices and data-set used in case needed
        print(self.config["data"]["labelled"])
        print(self.data_idx["labelled"])

    def train_dataloader(self):
        """Batch unlabelled and labelled data together"""

        l_batch_size = self.config["batch_size"]

        # Calculate larger batch size for unlabelled data
        u_batch_size = int(self.config["mu"] * l_batch_size)

        # Define loaders with different batch sizes
        loader_labelled = DataLoader(self.data["labelled"], l_batch_size, shuffle=True)
        loader_unlabelled = DataLoader(self.data["unlabelled"], u_batch_size, shuffle=True)

        # "Zip" loaders together for simultaneous loading
        loaders = {"unlabelled": loader_unlabelled, "labelled": loader_labelled}
        # dict of dataloaders
        combined_loaders = CombinedLoader(loaders, "min_size")  
        # see also max_size_cycle. 
        # Should each epoch be the size of the number of batches in smallest dataloader or the largest dataloader. 
        # Min size easier to compare epochs as cutting the unlabelled data by dif amounts, making metrics tricky to compare.
        return combined_loaders

    def val_dataloader(self):
        val_batch_size = 100
        loader_val = DataLoader(self.data["val"], val_batch_size)
        return loader_val

    def test_dataloader(self):
        """Batch test and unlabelled data sequentially"""
        u_batch_size = 200
        # test step where idx is an arg, list-type. Runs sequentially (all of one, than all of the next), while CombinedLoader does 
        loader_test = DataLoader(self.data["test"], int(len(self.data["test"])))
        loader_unlabelled = DataLoader(self.data["unlabelled"], u_batch_size)
        return [loader_test, loader_unlabelled]

    def save_hparams(self):
        self.hyperparams.update(
            {
                "n_labelled": len(self.data["labelled"]),
                "n_unlabelled": len(self.data["unlabelled"]),
                "n_test": len(self.data["test"]),
                "n_val": len(self.data["val"]),
                "f_u": label_fraction(self.data["unlabelled"], 0),
                "f_l": label_fraction(self.data["labelled"], 0),
            }
        )


def default_transforms(config, mu, sig):
    return {
        "weak": T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(180),
                T.ToTensor(),
                Circle_Crop(),
                T.Normalize((mu,), (sig,)),
            ]
        ),
        "unlabelled": TransformFixMatch(config, mu, sig),
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
