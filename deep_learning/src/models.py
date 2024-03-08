import hydra
import pytorch_lightning as pl
import torch
from torch import nn
from solvers import VelocityVerlet
from networks.basics import FrictionBlock, LambdaLayer

from omegaconf import DictConfig


class BaseLitModel(pl.LightningModule):
    """
    Base lightning model. 

    Reference: 
    https://github.com/gorodnitskiy/yet-another-lightning-hydra-template/blob/main/src/modules/components/lit_module.py
    """
    def __init__(
            self, 
            optimizer: DictConfig, 
            scheduler: DictConfig, 
            weight_init: str
        ) -> None:
        super(BaseLitModel, self).__init__()
        
        self.opt_params = optimizer
        self.slr_params = scheduler
        self.weight_init = weight_init

        self.training_step_outputs = [] 
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _init_weights(self):
        weight_init_fn = {
            "xavier_uniform": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_,
            }[self.weight_init]

        def init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                weight_init_fn(m.weight)

        self.apply(init_weights)

    def configure_optimizers(self):
        optimizer: torch.optim = hydra.utils.instantiate(
            self.opt_params, params=self.parameters(), _convert_="partial"
        )
        if self.slr_params.get("scheduler"):
            scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
                self.slr_params.scheduler,
                optimizer=optimizer,
                _convert_="partial",
            )
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.slr_params.get("extras"):
                for key, value in self.slr_params.get("extras").items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return {"optimizer": optimizer}
    

class BaseSolutionMap(BaseLitModel):
    """Base solution map model."""

    def __init__(
            self,
            Delta_t: float,
            problem: DictConfig,
            loss: DictConfig,
            regularization: DictConfig,
            use_dimensionless_for_loss: bool = True,
            **kwargs
        ) -> None:
        super(BaseSolutionMap, self).__init__(**kwargs)
        
        self.Delta_t = Delta_t

        self.problem = hydra.utils.instantiate(problem)
        self.compute_H = LambdaLayer(self.problem.compute_Hamiltonian, "compute_Hamiltonian")                            
        self.compute_Lagr = LambdaLayer(self.problem.compute_Lagrangian, "compute_Lagrangian")
        self.fine_solver_dt = VelocityVerlet(self.problem.compute_ddx, Delta_t/128, 8)
        self.use_dimensionless = True if "nondimensionalize" in dir(self.problem) else False
        if self.use_dimensionless:
            self.nondimensionalize = LambdaLayer(self.problem.nondimensionalize, "nondimensionalize")
            self.dimensionalize = LambdaLayer(self.problem.dimensionalize, "dimensionalize")
        self.use_dimensionless_for_loss = self.use_dimensionless and use_dimensionless_for_loss

        self.loss_fn: nn.Module = hydra.utils.instantiate(loss)
        self.comm_strength = regularization.get("comm_strength", 0.)
        self.lagr_strength = regularization.get("lagr_strength", 0.)
        self.seq_weights = None

    def set_seq_weights(self, weights):
        self.seq_weights = weights.to(self.dtype).to(self.device)

    def forward(self, u0, sequence_len):
        pass 
    
    def forward_1step(self, u):
        pass

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0, sequence_len=1001):
        u0 = batch
        pred_seq = self(u0, sequence_len)
        # pred_seq.insert(0, u0) 
        return pred_seq

    def model_step(self, batch, batch_idx, stage=None):
        u0 = batch[0]
        true_seq = batch
        pred_seq = self(u0, len(self.seq_weights))

        # loss
        loss, losses = self.calc_misfit_loss(pred_seq, true_seq)
        # loss_comm = self.calc_comm_loss(u0)
        # loss += self.comm_strength * loss_comm
        # loss_lagr = self.calc_lagr_loss(u0)
        # loss += self.lagr_strength * loss_lagr

        # metrics
        traj_errors = self.calc_traj_errors(pred_seq, true_seq)
        H_errors = self.calc_H_errors_fast(pred_seq, u0)

        metrics = {"loss": loss.detach()}
        for t in range(len(self.seq_weights)):
            metrics[f"loss_step{t}"] = losses[t].detach()
            metrics[f"traj_err_step{t}"] = traj_errors[t].detach()
            metrics[f"H_err_step{t}"] = H_errors[t].detach()
        # metrics["loss_comm"] = loss_comm.detach()
        # metrics["loss_lagr"] = loss_lagr.detach()
        batch_size = len(u0)

        if stage == "train":
            self.training_step_outputs.append({"batch_size": batch_size, "metrics": metrics})
            self.log("step_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        elif stage == "val": 
            self.validation_step_outputs.append({"batch_size": batch_size, "metrics": metrics})
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        elif stage == "test": 
            self.test_step_outputs.append({"batch_size": batch_size, "metrics": metrics})

        return {"loss": loss, "batch_size": batch_size, "metrics": metrics}

    def calc_misfit_loss(self, pred_seq, true_seq):    
        losses = torch.zeros(len(self.seq_weights)).to(pred_seq[0])

        for t in range(len(self.seq_weights)):
            ut_pred = pred_seq[t]
            ut_true = true_seq[t]
            if self.seq_weights[t] != 0:
                if self.use_dimensionless_for_loss:
                    ut_pred = self.nondimensionalize(ut_pred)
                    ut_true = self.nondimensionalize(ut_true)
                losses[t] = self.loss_fn(ut_pred, ut_true)

        misfit_loss = losses @ self.seq_weights / torch.sum(self.seq_weights)
        return misfit_loss, losses
    
    def calc_comm_loss(self, u0):
        phi_F_u0 = self(self.fine_solver_dt(u0), 2)[1]
        F_phi_u0 = self.fine_solver_dt(self(u0, 2)[1])
        if self.use_dimensionless_for_loss:
            phi_F_u0 = self.nondimensionalize(phi_F_u0)
            F_phi_u0 = self.nondimensionalize(F_phi_u0)
        return self.loss_fn(F_phi_u0, phi_F_u0)
    
    def calc_lagr_loss(self, u0):
        lagr_loss = 0.
        u = u0
        for _ in range(4):
            phi_u = self(u, 2)[1]
            lagr_loss += self.compute_Lagr(phi_u).mean() * self.fine_solver_dt.T
            u = self.fine_solver_dt(u)
        return lagr_loss 
    
    def calc_det_loss(self, u0):
        d = torch.stack([torch.det(torch.autograd.functional.jacobian(self.forward_1step, _u0, create_graph=True)) for _u0 in u0])
        return torch.mean((d-1)**2)
        
    def calc_traj_errors(self, pred_seq, true_seq):
        traj_errors = []
        for ut_pred, ut_true in zip(pred_seq, true_seq):
            if self.use_dimensionless_for_loss:
                ut_pred = self.nondimensionalize(ut_pred)
                ut_true = self.nondimensionalize(ut_true)
            # calculate mean l2-norm (NOT mean squared l2-norm) 
            diff_squares = nn.functional.mse_loss(ut_pred, ut_true, reduction="none")
            errors = diff_squares.sum(dim=-1).sqrt() / torch.sum(ut_true**2, dim=-1).sqrt()
            traj_errors.append(errors.mean())
        return traj_errors

    def calc_H_errors(self, pred_seq, true_seq):
        return [nn.functional.l1_loss(self.compute_H(ut_pred), self.compute_H(ut_true)) for ut_pred, ut_true in zip(pred_seq, true_seq)]

    def calc_H_errors_fast(self, pred_seq, u0):
        H_errors = []
        H0 = self.compute_H(u0)
        for ut_pred in pred_seq:
            diffs = nn.functional.l1_loss(self.compute_H(ut_pred), H0, reduction="none")
            errors = diffs / torch.abs(H0)
            H_errors.append(errors.mean())
        return H_errors


class SolutionMap(BaseSolutionMap):
    """Solution map."""

    def __init__(self, network: DictConfig, **kwargs):
        super(SolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        self.i2h = hydra.utils.instantiate(network.i2h)
        self.h2h = hydra.utils.instantiate(network.h2h)
        self.h2o = hydra.utils.instantiate(network.h2o)
        # self.friction = FrictionBlock()

        if self.weight_init is not None:
            self._init_weights()

    def forward_1step(self, u, return_hidden=False):
        if self.use_dimensionless:
            u = self.nondimensionalize(u)

        if return_hidden:
            u = self.i2h(u)
            u, hs = self.h2h(u, return_hidden=True)
            u = self.h2o(u)
            if self.use_dimensionless:
                u = self.dimensionalize(u)
            return u, hs
        else:
            u = self.i2h(u)
            u = self.h2h(u)
            u = self.h2o(u)
            if self.use_dimensionless:
                u = self.dimensionalize(u)
            return u
    
    def forward(self, u0, sequence_len):
        res = []

        if self.use_dimensionless:
            u0 = self.nondimensionalize(u0)
        hidden = self.i2h(u0)
        out = self.h2o(hidden)
        if self.use_dimensionless:
            out = self.dimensionalize(out)
        res.append(out)

        for _ in range(sequence_len-1):
            hidden = self.h2h(hidden)
            out = self.h2o(hidden)
            # out = out + self.friction(out)
            if self.use_dimensionless:
                out = self.dimensionalize(out)
            res.append(out)

        return res
    
    def freeze_encoder_decoder(self):
        for param in self.i2h.parameters():
            param.requires_grad = False
        for param in self.h2o.parameters():
            param.requires_grad = False


class CorrectionOperator(BaseSolutionMap):
    """Correction operator."""
    
    def __init__(self, coarse_h: float, network: DictConfig, **kwargs):
        super(CorrectionOperator, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        self.coarse_solver = VelocityVerlet(lambda x: self.problem.compute_ddx(x), self.Delta_t, int(self.Delta_t//coarse_h))
        self.net = hydra.utils.instantiate(network)

        if self.weight_init is not None:
            self._init_weights()

    def forward_1step(self, u):
        return self.net(self.coarse_solver(u))
    
    def forward(self, u0, sequence_len):
        res = []
        u = u0 
        for _ in range(sequence_len):
            u = self.forward_1step(u)
            res.append(u)
        return res
    
    
class CorrectionOperator2(BaseSolutionMap):
    def __init__(self, coarse_h: float, network: DictConfig, **kwargs):
        super(CorrectionOperator2, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)

        self.coarse_solver = VelocityVerlet(lambda x: self.problem.compute_ddx(x), self.Delta_t, int(self.Delta_t//coarse_h))
        self.net = hydra.utils.instantiate(network)

        if self.weight_init is not None:
            self._init_weights()
        
    def forward_1step(self, u):
        C_u = self.coarse_solver(u)
        return C_u + self.net(C_u)
        # return C_u + self.net(u)
    
    def forward(self, u0, sequence_len):
        res = []
        u = u0 
        for _ in range(sequence_len):
            u = self.forward_1step(u)
            res.append(u)
        return res


if __name__ == "__main__":

    from omegaconf import OmegaConf
    from utils.benchmark_utils import time_forward, time_backward, outputs_stats

    with hydra.initialize(version_base="1.3", config_path="../configs"):
        
        # compose default config and instantiate lightning module 
        cfg = hydra.compose(config_name="train", 
                            overrides=["experiment=fpu", "module/network=enc-resblocks-dec", 
                                    #    "module.network.h2h.n_linears_per_block=1",
                                    #    "module.network.h2h.n_blocks=3"
                                       ])
        # print(print(OmegaConf.to_yaml(cfg.module)))
        model = hydra.utils.instantiate(cfg.module, _recursive_=False)
        
        print(model)
        print("n_trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("device:", model.device)
        print("dtype:", model.dtype)
        
        # benchmark forward time 
        compare = time_forward(model, nsteps_list=[0, 1, 5])
        print(compare)
        
        # benchmark backward time 
        compare = time_backward(model, nsteps_list=[0, 1, 5])
        print(compare)
        
        # benchmark outputs stats 
        stats = outputs_stats(model, nsteps=5)
        print(stats)
