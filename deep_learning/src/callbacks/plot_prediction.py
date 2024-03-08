import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from typing import List
from torch import Tensor
from models import BaseSolutionMap


def reshape_predictions(predictions: List[Tensor]) -> List[Tensor]:
    traj_len = len(predictions)
    n_traj = len(predictions[0])

    trajectories = []
    for i in range(n_traj):
        trajectories.append(torch.stack([predictions[j][i] for j in range(traj_len)]))
    
    return trajectories


def plot_energy_profile(trajectory: Tensor, model: BaseSolutionMap, filepath: str = "", title: str = ""):

    t = np.arange(len(trajectory)) * model.Delta_t
    quantities = model.problem.compute_quantities(trajectory)
    quantities = {key: quantities[key].detach().cpu().numpy() for key in quantities.keys()}
    init_vals = [quantities[key][0] for key in quantities.keys()]
    min_val = min(init_vals)
    max_val = max(init_vals)
    val_range = max_val - min_val

    fig, ax = plt.subplots()
    for key in quantities.keys():
        ax.plot(t, quantities[key], linewidth=2, label=key)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(min_val - 0.1*val_range, max_val + 0.1*val_range)
    # ax.set_ylim(0, 1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("energy")
    ax.set_title(title)
    ax.text(0.95, 0.01, title,
        verticalalignment="bottom", horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=15)  # workaround for showing figure title in wandb panel 
    ax.legend()

    if filepath:
        plt.savefig(filepath, dpi=150)

    return fig


def plot_trajectory_argoncrystal(trajectory: Tensor, filepath: str = "", title: str = ""):

    x = trajectory[:, 14:].detach().cpu().numpy()
    # c = np.arange(len(trajectory))
    markers = [".", "^", "s", "o", "*", "+", "h"]

    fig, ax = plt.subplots()
    for i in range(7):
        x1 = x[:, i*2]
        x2 = x[:, i*2+1]
        ax.scatter(x1, x2, s=2, marker=markers[i])
    for i in range(7):
        x1 = x[0, i*2]
        x2 = x[0, i*2+1]
        ax.scatter(x1, x2, c="r", s=4, marker=markers[i])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.text(0.95, 0.01, title,
        verticalalignment="bottom", horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=15)  # workaround for showing figure title in wandb panel 
    ax.set_aspect("equal")
    # plt.grid()

    if filepath:
        plt.savefig(filepath, dpi=150)

    return fig


class PlotEnergyProfile(pl.Callback):

    def __init__(self, nsteps: int = 100, log_freq: int = 2) -> None:
        self.nsteps = nsteps
        self.log_freq = log_freq

    def setup(self, trainer: pl.Trainer, pl_module: BaseSolutionMap, stage: str) -> None:
        self.predict_samples: Tensor = pl_module.problem.default_initial_states()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch % self.log_freq == 0:
            self.predict_and_plot(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        self.predict_and_plot(trainer, pl_module)
    
    def predict_and_plot(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        predict_samples = self.predict_samples.to(pl_module.dtype).to(pl_module.device)
        predictions = pl_module.predict_step(predict_samples, batch_idx=0, sequence_len=self.nsteps)
        trajectories = reshape_predictions(predictions)

        log = {}
        for i, traj in enumerate(trajectories):
            fig = plot_energy_profile(traj, pl_module, title=f"epoch {trainer.current_epoch}")
            log[f"predict/sample_{i+1}"] = fig
        trainer.logger.experiment.log(log, commit=False)


class PlotTrajectory(pl.Callback):

    def __init__(self, nsteps: int = 100, log_freq: int = 2) -> None:
        self.nsteps = nsteps
        self.log_freq = log_freq

    def setup(self, trainer: pl.Trainer, pl_module: BaseSolutionMap, stage: str) -> None:
        self.predict_samples: Tensor = pl_module.problem.default_initial_states()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        if trainer.sanity_checking:  # optional skip
            return
        if trainer.current_epoch % self.log_freq == 0:
            self.predict_and_plot(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        self.predict_and_plot(trainer, pl_module)
    
    def predict_and_plot(self, trainer: pl.Trainer, pl_module: BaseSolutionMap) -> None:
        predict_samples = self.predict_samples.to(pl_module.dtype).to(pl_module.device)
        predictions = pl_module.predict_step(predict_samples, batch_idx=0, sequence_len=self.nsteps)
        trajectories = reshape_predictions(predictions)

        log = {}
        for i, traj in enumerate(trajectories):
            fig = plot_trajectory_argoncrystal(traj, title=f"epoch {trainer.current_epoch}")
            log[f"predict/sample_{i+1}_xspace"] = fig
        trainer.logger.experiment.log(log, commit=False)

  
