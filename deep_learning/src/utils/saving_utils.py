import logging
import os
import pandas as pd
import torch 
from callbacks.plot_prediction import plot_energy_profile, plot_trajectory_argoncrystal

from typing import List
from torch import Tensor
from models import BaseSolutionMap


logger = logging.getLogger(__name__)


def save_metrics(metrics: dict, dirname: str) -> None:
    """Save metrics dict returned by `Trainer.test` method."""

    path = os.path.join(dirname, f"test_metrics.csv")
    df = pd.DataFrame.from_dict(metrics).T
    df.to_csv(path)
    
    logger.info(f"Saved test metrics to: {path}")


def save_predictions(predictions: List[List[Tensor]], dirname: str) -> None:
    """Save predictions returned by `Trainer.predict` method."""

    if not predictions:
        logger.warning("Predictions is empty! Saving was cancelled ...")
        return

    path = os.path.join(dirname, "predictions")
    os.makedirs(path, exist_ok=True)

    n_traj = len(predictions)
    traj_len = len(predictions[0])
    dof = len(predictions[0][0]) // 2

    cols = [f"v{i}" for i in range(1, dof+1)] + [f"x{i}" for i in range(1, dof+1)]

    for i in range(n_traj):
        data = torch.stack(predictions[i]).numpy()
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(os.path.join(path, f"traj{i+1}.csv"), index=False)
    
    logger.info(f"Saved {n_traj} predicted trajectories (traj_len = {traj_len}) to: {path}")


def save_energy_plots(predictions: List[List[Tensor]], model: BaseSolutionMap, dirname: str) -> None:
    """Plot and save energy profiles computed from predictions returned by `Trainer.predict` method."""

    if not predictions:
        logger.warning("Predictions is empty! Saving was cancelled ...")
        return

    path = os.path.join(dirname, "predictions")
    os.makedirs(path, exist_ok=True)

    n_traj = len(predictions)

    for i in range(n_traj):
        data = torch.stack(predictions[i])
        plot_energy_profile(data, model, filepath=os.path.join(path, f"traj{i+1}.pdf"))
        # plot_trajectory_argoncrystal(data, filepath=os.path.join(path, f"x_traj{i+1}.pdf"))
            
    logger.info(f"Saved {n_traj} energy plots to: {path}")
