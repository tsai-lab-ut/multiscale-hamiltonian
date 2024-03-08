import torch
from pytorch_lightning.callbacks import Callback


class ManualMetricsLogger(Callback):
    
    def on_train_epoch_end(self, trainer, pl_module):
        self._shared_epoch_end(trainer, pl_module, "train")
        pl_module.training_step_outputs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        self._shared_epoch_end(trainer, pl_module, "val")
        pl_module.validation_step_outputs.clear()
    
    def on_test_epoch_end(self, trainer, pl_module):
        self._shared_epoch_end(trainer, pl_module, "test")
        pl_module.test_step_outputs.clear()
    
    def _shared_epoch_end(self, trainer, pl_module, stage=None):
        if stage == "train":
            outputs = pl_module.training_step_outputs
        elif stage == "val":
            outputs = pl_module.validation_step_outputs
        elif stage == "test":
            outputs = pl_module.test_step_outputs
        
        n_total = sum([out["batch_size"] for out in outputs])
        
        def aggregate_outputs(m):
            return torch.stack([out["metrics"][m] * out["batch_size"] for out in outputs]).sum() / n_total
        
        metrics = {m: aggregate_outputs(m) for m in outputs[0]["metrics"].keys()}
        logs = dict()
        logs[f"{stage}/batch_size"] = torch.tensor(outputs[0]["batch_size"], dtype=torch.float32)
        for m in metrics.keys():
            logs["/".join([stage, m])] = metrics[m].detach()
        if trainer.is_global_zero:
            pl_module.log_dict(logs, sync_dist=True, rank_zero_only=True)
