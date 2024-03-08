import hydra
import logging
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from utils.utils import (
    instantiate_callbacks,
    instantiate_litloggers,
    get_run_name,
    get_run_id
)
from utils.saving_utils import save_metrics, save_predictions, save_energy_plots
from utils.benchmark_utils import time_forward

from omegaconf import DictConfig
from typing import List


logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
    "get_run_name", lambda ckpt_path: get_run_name(ckpt_path), use_cache=True
)
OmegaConf.register_new_resolver(
    "get_run_id", lambda ckpt_path: get_run_id(ckpt_path), use_cache=True
)

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> pl.Trainer:

    assert cfg.ckpt_path

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

    # Load lightning model
    logger.info(f"Loading lightning model <{cfg.module._target_}> from checkpoint {cfg.ckpt_path}")
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.module, **checkpoint["hyper_parameters"], _recursive_=False)
    if cfg.datamodule.dtype == "float64":
        model = model.double()
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    logger.info(f"{model}")

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
    
    # Test the model
    logger.info("Starting testing!")
    metrics = trainer.test(
        model=model, 
        datamodule=datamodule
    )
    save_metrics(metrics, dirname=cfg.paths.output_dir)

    # Benchmark forward time
    logger.info("Starting benchmarking forward time!")
    t_forward = time_forward(model)
    logger.info(t_forward)

    # Make predictions
    if cfg.get("predict"):
        logger.info("Starting predicting!")
        predict_samples = model.problem.default_initial_states()
        predict_samples = predict_samples.to(model.dtype).to(model.device)
        logger.info(f"Predict samples = {predict_samples}")
        predictions = trainer.predict(
            model=model,
            dataloaders=predict_samples,
        )
        save_predictions(
            predictions=predictions,
            dirname=cfg.paths.output_dir
        )
        save_energy_plots(
            predictions=predictions,
            model=model,
            dirname=cfg.paths.output_dir
        )

    return trainer 


if __name__ == "__main__":
    main()
