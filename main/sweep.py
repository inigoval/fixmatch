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
    project=f"mirabest-sweep-{config['type']}",
    save_dir=path_dict["files"],
    config=config,
)

config = wandb_logger.experiment.config

print(config)

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

model = clf(config)

# Train model #
trainer.fit(model, data)
trainer.test(ckpt_path="best")

wandb_logger.experiment.finish()
