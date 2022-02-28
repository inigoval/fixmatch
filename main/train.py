import wandb
import torch
import numpy as np
import pytorch_lightning as pl

from paths import Path_Handler
from callbacks import MetricLogger, FeaturePlot, ImpurityLogger
from dataloading.datamodules import mbDataModule, GalaxyMNISTDataModule
from fixmatch import clf
from config import load_config, update_config
from utilities import log_examples

config = load_config()

paths = Path_Handler()  # paths in project directory - nice, inigo
path_dict = paths._dict()

# pick seed i and seed f, will go between 
for s in range(config["seed_i"], config["seed_f"]):

    config["seed"] = s
    pl.seed_everything(s)

    # Save model with best accuracy for test evaluation, model will be saved in wandb and also #
    dirpath = f"wandb/{config['data']}_{config['method']}_{config['data_source']}_{config['data']['unlabelled']}"
    if config['data_source'] == 'rgz':
        dirpath += f"_cut{config['cut_threshold']}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/accuracy",
        mode="max",
        every_n_epochs=3,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=True,
        verbose=True,
        dirpath=dirpath,
        filename=f"seed{config['seed']}",
        save_weights_only=True,
    )

    # Initialise wandb logger, change this if you want to use a different logger #
    wandb_logger = pl.loggers.WandbLogger(
        project=config["project_name"],
        save_dir=path_dict["files"],
        reinit=True,
        config=config,
    )

    # Load data and record hyperparameters #
    # data = mbDataModule(config)  # mb = MiraBest. Also includes hparams?
    data = GalaxyMNISTDataModule(config)
    data.prepare_data()
    data.setup()
    # exit()
    wandb_logger.log_hyperparams(data.hyperparams)
    log_examples(wandb_logger, data.data["unlabelled"])

    # Record mean and standard deviation used in normalisation for inference #
    config["data"]["mu"] = data.mu.item()
    config["data"]["sig"] = data.sig.item()

    # you can add ImpurityLogger if NOT using rgz unlabelled data to track impurities and mask rate
    # ImpurityLogger will track the masking rate i.e. how many labels get propogated (controlled by tau, confidence threshold above which model uses the label e.g. 0.05)
    # and the impurity rate i.e. how many propogated labels are erroneous. 
    callbacks = {
        "baseline": [MetricLogger(), checkpoint_callback],
        # "fixmatch": [MetricLogger(), ImpurityLogger(), checkpoint_callback],
        "fixmatch": [MetricLogger(), checkpoint_callback],
    }

    trainer = pl.Trainer(
        # gpus=1,
        max_epochs=config["train"]["n_epochs"],
        logger=wandb_logger,
        deterministic=True,
        callbacks=callbacks[config["method"]],
        check_val_every_n_epoch=3,
        log_every_n_steps=10,
    )

    # Initialise model #
    model = clf(config)

    # Train model #
    trainer.fit(model, data)

    # Print label indices #
    print("labelled data indices: ", data.data_idx["labelled"])

    # Run test loop #
    trainer.test(model, dataloaders=data, ckpt_path="best")

    # Save model in wandb #
    wandb.save(checkpoint_callback.best_model_path)

    wandb_logger.experiment.finish()

# mu - hyperparameter controlling weighting (in practice, multiplier of unlabelled batch size vs labelled batch size)

# dataloaders get zipped together
