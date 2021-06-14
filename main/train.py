import wandb
import torch
import numpy as np
import pytorch_lightning as pl

from paths import Path_Handler
from callbacks import MetricLogger, FeaturePlot
from dataloading.datamodules import mbDataModule, mb_rgzDataModule
from fixmatch import clf
from config import load_config

config = load_config()
paths = Path_Handler()
path_dict = paths._dict()

# n_epochs = config["train"]["n_epochs"]
# Normalise epoch number to account for data splitting
# n_epochs = int(config["train"]["n_epochs"] / config["data"]["fraction"])


# Save model with best accuracy for test evaluation #
# checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val/accuracy", mode="max")

for seed in range(10):

    ## Set seeds for reproducibility ##
    pl.seed_everything(seed)

    # Initialise wandb logger and save hyperparameters
    wandb_logger = pl.loggers.WandbLogger(
        project="mirabest-classification",
        save_dir=path_dict["files"],
        reinit=True,
        config=config
        #        mode="disabled",
    )

    # config = wandb_logger.experiment.config

    wandb_logger.log_hyperparams(
        {"data": {"split": config["data"]["split"]}, "seed": seed}
    )

    config["seed"] = seed

    data_modules = {
        "mirabest": mbDataModule(config),
        "rgz": mb_rgzDataModule(config),
    }

    # Load data #
    data = data_modules[config["dataset"]]
    data.prepare_data()
    data.setup()
    wandb_logger.log_hyperparams(data.hparams)

    # wandb_logger.experiment.config["train"]["n_epochs"] = int(
    #    config["train"]["n_epochs"] / (data.u_frac)
    # )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config["train"]["n_epochs"],
        logger=wandb_logger,
        deterministic=True,
        callbacks=[MetricLogger()],
    )

    model = clf(config)

    # Train model #
    trainer.fit(model, data)
    trainer.test()

    wandb_logger.experiment.finish()
