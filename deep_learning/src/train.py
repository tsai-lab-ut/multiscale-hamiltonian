import hydra
import logging
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from utils.utils import (
    instantiate_callbacks,
    instantiate_litloggers,
    log_hyperparameters,
    get_run_name,
    get_run_id
)
from callbacks.wandb_callbacks import get_wandb_logger

from omegaconf import DictConfig
from typing import List


logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "get_run_name", lambda ckpt_path: get_run_name(ckpt_path), use_cache=True
)
OmegaConf.register_new_resolver(
    "get_run_id", lambda ckpt_path: get_run_id(ckpt_path), use_cache=True
)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> pl.Trainer:

    # Print package versions
    logger.info(f"Using pytorch {torch.__version__}")
    logger.info(f"Using pytorch lightning {pl.__version__}")
    logger.info(f"Using hydra {hydra.__version__}")

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        logger.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    logger.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Init lightning model
    logger.info(f"Instantiating lightning model <{cfg.module._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.module, _recursive_=False)
    if cfg.datamodule.dtype == "float64":
        model = model.double()
    logger.info(f"{model}")

    # Load trained model if given ckpt path
    if cfg.get("init_model_ckpt"):
        logger.info(f"Loading model state dict from {cfg.init_model_ckpt}")
        checkpoint = torch.load(cfg.init_model_ckpt)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Freeze part of model 
    if cfg.get("freeze_encoder_decoder"):
        model.freeze_encoder_decoder()

    # Init callbacks
    logger.info("Instantiating callbacks...")
    callbacks: List[pl.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Init lightning loggers
    logger.info("Instantiating lightning loggers...")
    litloggers: List[pl.loggers.Logger] = instantiate_litloggers(cfg.get("loggers"))

    # Init profiler
    if cfg.get("profiler"):
        logger.info(f"Instantiating profiler <{cfg.profiler._target_}>")
        profiler: pl.profilers.Profiler = hydra.utils.instantiate(cfg.profiler)
    else:
        profiler = None

    # Init lightning trainer
    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=litloggers, profiler=profiler)

    # Log hyperparameters
    if litloggers:
        logger.info("Logging hyperparameters!")
        log_hyperparameters(
            {
                "cfg": cfg,
                "model": model,
                "trainer": trainer,
            }
        )
    
    # Log gradients and model topology with wandb logger 
    wandb_logger = get_wandb_logger(trainer)
    if wandb_logger:
        wandb_logger.watch(model, log="all", log_freq=500, log_graph=True)

    # Train the model
    logger.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.get("resume_from_ckpt"),
    )

    # Test the model
    if cfg.get("test"):
        logger.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            logger.warning(
                "Best ckpt not found! Using current weights for testing..."
            )
            ckpt_path = None
        trainer.test(
            model=model, 
            datamodule=datamodule, 
            ckpt_path=ckpt_path
        )
        logger.info(f"Best ckpt path: {ckpt_path}")

    return trainer 


if __name__ == "__main__":
    main()
