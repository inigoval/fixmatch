import wandb
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
import torchmetrics.functional as tmF

from utilities import fig2img, entropy, dset2tens, batch_eval
from config import load_config


config = load_config()


class MetricLogger(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_test_epoch_end(self, trainer, pl_module):

        x, y = dset2tens(pl_module.trainer.datamodule.data["test"])
        x = x.type_as(pl_module.C.conv1[0].weight)
        y = y.type_as(pl_module.C.conv1[0].weight)

        y_hat = pl_module.forward(x)

        # Softmax entropy #
        H = entropy(y_hat)
        H = H.tolist()

        # Plot histogram #
        data = [[h, pl_module.current_epoch] for h in H]
        table = wandb.Table(data=data, columns=["entropy", "epoch"])
        trainer.logger.experiment.log(
            {
                "test/entropy": wandb.plot.histogram(
                    table, "entropy", title="softmax entropy (test)"
                )
            }
        )

        y_hat = pl_module.forward(x)
        p_pred, y_pred = torch.max(y_hat, 1)
        idx_wrong = torch.nonzero(y_pred != y).view(-1)

        H_wrong = np.mean(entropy(y_hat[idx_wrong, ...]))
        pl_module.log("average misclassification entropy", H_wrong)
        # pl_module.log("test/average misclassification probability", torch.mean(p_pred))

        # Plot bar chart of incorrect indices #


#        data = [[idx + 0.1] for idx in idx_wrong.tolist()]
#        table = wandb.Table(data=data, columns=["index"])
#        trainer.logger.experiment.log(
#            {
#                "test/misclassifications": wandb.plot.histogram(
#                    table, "index", title="misclassifications"
#                )
#            }
#        )


class FeaturePlot(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_test_epoch_end(self, trainer, pl_module):
        x_l, y_l = dset2tens(pl_module.trainer.datamodule.data["l"])
        x_l = x_l.type_as(pl_module.C.conv1[0].weight)
        y_l = y_l.type_as(pl_module.C.conv1[0].weight)

        x_test, y_test = dset2tens(pl_module.trainer.datamodule.data["test"])
        x_test = x_test.type_as(pl_module.C.conv1[0].weight)
        y_test = y_test.type_as(pl_module.C.conv1[0].weight)

        x_u, y_u = dset2tens(pl_module.trainer.datamodule.data["u"])[0]
        x_u = x_u.type_as(pl_module.C.conv1[0].weight)
        y_u = y_u.type_as(pl_module.C.conv1[0].weight)

        plot_dict = {}

        for x, y, name in zip(
            [x_u, x_test, x_l],
            [y_u, y_test, y_l],
            ["unlabelled", "test", "labelled"],
        ):

            logits = pl_module.forward(x, logit=True)

            data = torch.cat((logits, y.view(-1, 1)), 1).tolist()

            table = wandb.Table(data=data, columns=["fr1", "fr2", "label"])

            plot_dict[f"test/logits {name}"] = wandb.plot.scatter(
                table, "fr1", "fr2", title=f"{name} logits"
            )

        trainer.logger.experiment.log(plot_dict)


class ImpurityLogger(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):

        # Initialise metrics into dictionary
        self.fn_dict = {
            "impurities": self.Impurities(pl_module),
            "mask_rate": self.MaskRate(pl_module),
        }
        data_u = pl_module.trainer.datamodule.data["u"]

        outs = batch_eval(self.fn_dict, data_u)

        mask_rate = np.sum(np.asarray(outs["mask_rate"])) / len(data_u)
        pl_module.log("train/mask_rate", mask_rate)

        impurities = np.sum(np.asarray(outs["impurities"])) / len(data_u)
        pl_module.log("train/impurities", impurities / (mask_rate + 0.0000001))

    #        tau = pl_module.config["tau"]
    #        x, y = dset2tens(pl_module.trainer.datamodule.data["u"])
    #
    #        y_hat = pl_module.forward(x)
    #        p_pred, y_pred = torch.max(y_hat, 1)
    #
    #        mask_rate = self.calc_mask_rate(tau, p_pred)
    #
    #        impurities = self.calc_impurities(tau, p_pred, y_pred, y)

    def on_test_epoch_end(self, trainer, pl_module):

        data_u = pl_module.trainer.datamodule.data["u"]

        outs = batch_eval(self.fn_dict, data_u)

        mask_rate = np.sum(np.asarray(outs["mask_rate"])) / len(data_u)
        pl_module.log("unlabelled/mask_rate", mask_rate)

        impurities = np.sum(np.asarray(outs["impurities"])) / len(data_u)
        pl_module.log("unlabelled/impurities", impurities / (mask_rate + 0.0000001))

    class Impurities:
        """
        Class that calculates impurities, for use in the batch_eval function
        """

        def __init__(self, pl_module):
            self.pl_module = pl_module

        def __call__(self, x, y):
            # Take only weakly transformed sample
            x = x[0]

            # Move data to correct memory
            x = x.type_as(self.pl_module.C.conv1[0].weight)
            y = y.type_as(self.pl_module.C.conv1[0].weight)

            y_hat = self.pl_module.forward(x)
            p_pred, y_pred = torch.max(y_hat, 1)

            idx_wrong = torch.nonzero(y_pred != y).view(-1)
            p_wrong = p_pred[idx_wrong, ...]
            impurities = torch.count_nonzero(p_wrong > self.pl_module.config["tau"])
            return impurities.cpu().detach().numpy()

    class MaskRate:
        """
        Class that calculates mask rate, for use in the batch_eval function
        """

        def __init__(self, pl_module):
            self.pl_module = pl_module

        def __call__(self, x, y):
            # Take only weakly transformed sample
            x = x[0]

            # Move data to correct memory
            x = x.type_as(self.pl_module.C.conv1[0].weight)
            y = y.type_as(self.pl_module.C.conv1[0].weight)

            y_hat = self.pl_module.forward(x)
            p_pred, y_pred = torch.max(y_hat, 1)

            masks = torch.count_nonzero(p_pred > self.pl_module.config["tau"])
            return masks.cpu().detach().numpy()

    @staticmethod
    def calc_impurities(tau, p_pred, y_pred, y):
        idx_wrong = torch.nonzero(y_pred != y).view(-1)
        p_wrong = p_pred[idx_wrong, ...]
        impurities = torch.count_nonzero(p_wrong > tau)
        return impurities / len(y)

    @staticmethod
    def calc_mask_rate(tau, p_pred):
        masks = torch.count_nonzero(p_pred > tau)
        return masks / len(p_pred)
