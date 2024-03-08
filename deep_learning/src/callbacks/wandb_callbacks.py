"""
Wandb callbacks

https://github.com/gorodnitskiy/yet-another-lightning-hydra-template/blob/main/src/callbacks/wandb_callbacks.py
"""

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    # if hasattr(trainer, "fast_dev_run") and trainer.fast_dev_run:
    #     raise Exception(
    #         "Cannot use wandb callbacks since pytorch lightning disables"
    #         "loggers in `fast_dev_run=true` mode."
    #     )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    # raise Exception(
    #     "You are using wandb related callback, but WandbLogger was not"
    #     "found for some reason..."
    # )
    return None


class WatchModel(Callback):   # Note: this callback fails to log parameters, only logs gradients 
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "all", log_freq: int = 500, log_graph: bool = False) -> None:
        self.log = log
        self.log_freq = log_freq
        self.log_graph = log_graph

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(
            model=pl_module,
            log=self.log,
            log_freq=self.log_freq,
            log_graph=self.log_graph,
        )
