import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.trainer.supporters import CombinedLoader

from config import load_config
from paths import Path_Handler
from dataloading.datasets import MB_nohybrids, RGZ20k, MiraBest_full
from dataloading.utils import Circle_Crop, label_fraction, flip_targets
from dataloading.utils import size_cut, mb_cut, data_splitter, subset
from fixmatch import TransformFixMatch

paths = Path_Handler()
path_dict = paths._dict()
config = load_config()

weak_T = T.Compose([T.RandomRotation(180), T.ToTensor(), Circle_Crop()])
u_T = TransformFixMatch()
# u_T = T.Compose([T.ToTensor(), Circle_Crop()])
test_T = T.Compose([T.ToTensor(), Circle_Crop()])


class mb_rgzDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.cut_threshold = config["cut_threshold"]
        self.fraction = config["data"]["fraction"]
        self.split = config["data"]["split"]
        self.val_frac = config["data"]["val_frac"]
        self.batch_size = config["train"]["batch_size"]
        self.path = path_dict["data"]
        self.hparams = {}
        self.config = config

    def prepare_data(self):
        MB_nohybrids(path_dict["mb"], train=True, download=True)
        MB_nohybrids(path_dict["mb"], train=False, download=True)
        RGZ20k(path_dict["rgz"], train=True, download=True)

    def setup(self, stage=None):
        # MiraBest test samples #

        self.data, self.data_idx = data_splitter(
            MB_nohybrids(path_dict["mb"], train=True, transform=weak_T),
            fraction=self.fraction,
            split=self.split,
            val_frac=self.val_frac,
        )
        self.data["test"] = MB_nohybrids(path_dict["mb"], train=False, transform=test_T)

        # Cut unlabelled RGZ samples and define unlabelled training set #
        rgz = RGZ20k(path_dict["rgz"], train=True, transform=u_T)
        self.size_cut(rgz)
        mb_cut(rgz)
        len_u = torch.clamp(
            torch.tensor(self.config["data"]["u_frac"] * len(self.data["l"])),
            0,
            len(self.data["u"]),
        ).item()
        rgz = subset(rgz, len_u)
        self.data["u"] = rgz

        # Flip a number of targets randomly #
        if config["train"]["flip"]:
            self.data["l"] = flip_targets(self.data["l"], config["train"]["flip"])

        self.save_hparams()

    def train_dataloader(self):
        loader_l = DataLoader(self.data["l"], self.batch_size, shuffle=True)
        loader_u = DataLoader(
            self.data["u"], self.batch_size * self.config["mu"], shuffle=True
        )
        loaders = {"u": loader_u, "l": loader_l}
        combined_loaders = CombinedLoader(loaders, "min_size")
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
        idx = np.argwhere(dset.sizes > self.cut_threshold).flatten()
        dset.data = dset.data[idx, ...]
        dset.names = dset.names[idx, ...]
        dset.rgzid = dset.rgzid[idx, ...]
        dset.sizes = dset.sizes[idx, ...]
        dset.mbflg = dset.mbflg[idx, ...]
        self.u_frac = idx.shape[0] / length


class mbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        path=path_dict["data"],
    ):
        super().__init__()
        self.path = path
        self.fraction = config["data"]["fraction"]
        self.split = config["data"]["split"]
        self.val_frac = config["data"]["val_frac"]
        self.batch_size = config["train"]["batch_size"]
        self.hparams = {}
        self.config = config

    def prepare_data(self):
        MB_nohybrids(self.path, train=True, download=True)
        MB_nohybrids(self.path, train=False, download=True)

    def setup(self, stage=None):
        self.data, self.data_idx = data_splitter(
            MB_nohybrids(self.path, train=True, transform=weak_T),
            fraction=self.fraction,
            split=self.split,
            val_frac=self.val_frac,
        )

        self.data["u"] = torch.utils.data.Subset(
            MB_nohybrids(self.path, train=True, transform=u_T), self.data_idx["u"]
        )

        self.data["test"] = MB_nohybrids(self.path, train=False, transform=test_T)

        # Flip a number of targets randomly
        if config["train"]["flip"]:
            self.data["l"] = flip_targets(self.data["l"], config["train"]["flip"])

        self.save_hparams()

    def train_dataloader(self):
        loader_l = DataLoader(self.data["l"], self.batch_size, shuffle=True)
        loader_u = DataLoader(
            self.data["u"], self.batch_size * self.config["mu"], shuffle=True
        )
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


class mb_rgz_multiclassDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fraction=config["data"]["fraction"],
        split=config["data"]["split"],
        batch_size=config["train"]["batch_size"],
    ):
        super().__init__()
        self.fraction = fraction
        self.split = split
        self.batch_size = batch_size
        self.hparams = {}

    def prepare_data(self):
        MiraBest_full(path_dict["mb"], train=True, download=True)
        MiraBest_full(path_dict["mb"], train=False, download=True)
        RGZ20k(path_dict["rgz"], train=True, download=True)

    def setup(self, stage=None):
        # MiraBest test samples #
        mb_test = MiraBest_full(
            path_dict["mb"], train=False, transform=test_transforms["mb"]
        )
        self.test = mb_test

        # Labelled MiraBest samples #
        mb_l = MiraBest_full(path_dict["mb"], train=True, transform=transforms["mb"])
        if self.fraction != 1:
            mb_l, _ = d_split(mb_l, self.fraction)
        self.train_l = mb_l

        # Cut unlabelled RGZ samples and define unlabelled training set #
        mb_u = RGZ20k(path_dict["rgz"], train=True, transform=transforms["mb"])
        size_cut(config["data"]["cut_threshold"], mb_u)
        mb_cut(mb_u)
        self.train_u = mb_u

        # Combine labelled and unlabelled datasets #
        self.train = torch.utils.data.ConcatDataset([mb_u, mb_l])

        # Flip a number of targets randomly #
        if config["train"]["flip"]:
            mb_l = flip_targets(mb_l, config["train"]["flip"])

        self.save_hparams()

    def train_dataloader(self):
        loader_l = DataLoader(self.train_l, self.batch_size, shuffle=True)
        loader_u = DataLoader(self.train_u, self.batch_size, shuffle=True)
        loader_real = DataLoader(self.train, self.batch_size, shuffle=True)
        loaders = {"u": loader_u, "l": loader_l, "real": loader_real}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders

    def val_dataloader(self):
        loader_test = DataLoader(self.test, int(len(self.test) / 10))
        loader_u = DataLoader(self.train_u, int(len(self.train_u) / 10))
        loaders = {
            "test": loader_test,
            # "u": loader_u,
        }
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders

    def save_hparams(self):
        self.hparams.update(
            {
                "n_labelled": len(self.train_l),
                "n_unlabelled": len(self.train_u),
                "n_test": len(self.test),
                "f_l": label_fraction(self.train_l, 0),
            }
        )


class mnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fraction=config["data"]["fraction"],
        split=config["data"]["split"],
        batch_size=50,
        path=path_dict["data"],
    ):
        super().__init__()
        self.fraction = fraction
        self.split = split
        self.batch_size = batch_size
        self.transform = transforms["mnist"]
        self.path = path

    def prepare_data(self):
        MNIST(self.path, train=True, download=True)
        MNIST(self.path, train=False, download=True)

    def setup(self, stage=None):
        mnist_train = MNIST(self.path, train=True, transform=self.transform)
        mnist_test = MNIST(self.path, train=False, transform=self.transform)
        self.test_dataset = mnist_test

        if self.fraction != 1:
            n = len(mnist_train)
            n_train = int(n * self.fraction)
            mnist_train, _ = random_split(mnist_train, [n_train, n - n_train])
        self.train_dataset = mnist_train

        n = len(mnist_train)
        n_l = int(n * self.split)
        mnist_l, mnist_u = random_split(mnist_train, [n_l, n - n_l])
        self.train_dataset_u = mnist_u
        self.train_dataset_l = mnist_l

    def train_dataloader(self):
        loader_l = DataLoader(self.train_dataset_l, self.batch_size)
        loader_u = DataLoader(self.train_dataset_u, self.batch_size)
        loader_real = DataLoader(self.train_dataset, self.batch_size)
        loaders = {"u": loader_u, "l": loader_l, "real": loader_real}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders

    def val_dataloader(self):
        loader_test = DataLoader(self.test_dataset, int(len(self.test_dataset) / 10))
        loader_u = DataLoader(self.train_dataset_u, int(len(self.train_dataset_u) / 10))
        loaders = {"u": loader_u, "test": loader_test}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders

    def save_hparams(self):
        self.hparams.update(
            {
                "n_labelled": len(self.train_l),
                "n_unlabelled": len(self.train_u),
                "n_test": len(self.test),
            }
        )
