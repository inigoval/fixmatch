import wandb
import torch
import numpy as np
import pytorch_lightning as pl

from paths import Path_Handler
from callbacks import MetricLogger, FeaturePlot, ImpurityLogger
from dataloading.datamodules import mbDataModule
from fixmatch import clf
from config import load_config, update_config

config = load_config()

paths = Path_Handler()
path_dict = paths._dict()

for s in range(10):

    pl.seed_everything(s)

    # Save model with best accuracy for test evaluation #
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/accuracy",
        mode="max",
        every_n_epochs=3,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=True,
        verbose=True,
    )

    # Initialise wandb logger and save hyperparameters
    wandb_logger = pl.loggers.WandbLogger(
        project="domain-shift",
        save_dir=path_dict["files"],
        reinit=True,
        config=config,
    )

    # Load data and record hyperparameters #
    data = mbDataModule(config)
    data.prepare_data()
    data.setup()
    wandb_logger.log_hyperparams(data.hyperparams)

    callbacks = {
        "baseline": [MetricLogger(), checkpoint_callback],
        "fixmatch": [MetricLogger(), ImpurityLogger(), checkpoint_callback],
    }

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config["train"]["n_epochs"],
        logger=wandb_logger,
        deterministic=True,
        callbacks=callbacks[config["type"]],
        check_val_every_n_epoch=3,
        log_every_n_steps=10,
    )

    # initialise model
    model = clf(config)

    # Train model #
    trainer.fit(model, data)

    # Print label indices #
    print("labelled data indices: ", data.data_idx["l"])

    # Run test loop #
    trainer.test(ckpt_path="best")

    # Save model in wandb #
    wandb.save(checkpoint_callback.best_model_path)

    wandb_logger.experiment.finish()
