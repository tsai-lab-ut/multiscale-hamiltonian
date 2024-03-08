from torch import nn
from networks.basics import LambdaLayer, Scaler
from torch import Tensor
from problems import SeparableHamiltonianSystem


class ScaledMSELoss(nn.Module):
    """MSE loss with scaled input and target."""
    
    def __init__(self, scaler: Scaler, reduction: str = "mean") -> None:
        super().__init__()

        self.scaler = scaler
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):
        return nn.functional.mse_loss(self.scaler(input), self.scaler(target), reduction=self.reduction)

    
class MeanEnergyNormSquaredLoss(nn.Module):
    """Mean energy norm squared loss."""

    def __init__(self, problem: SeparableHamiltonianSystem, reduction: str = "mean") -> None:
        super().__init__()

        self.transform: nn.Module = LambdaLayer(
            problem.transform_to_energy_components, 
            "transform_to_energy_components"
        )
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):
        input = self.transform(input)
        target = self.transform(target)
        return nn.functional.mse_loss(input, target, reduction=self.reduction)


class AnchoredMeanEnergyNormSquaredLoss(nn.Module):
    """Anchored mean energy norm squared loss."""

    def __init__(self, problem: SeparableHamiltonianSystem, reduction: str = "mean") -> None:
        super().__init__()

        self.transform: nn.Module = LambdaLayer(
            problem.transform_to_energy_components_anchored, 
            "transform_to_energy_components_anchored"
        )
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):
        input = self.transform(input)
        target = self.transform(target)
        return nn.functional.mse_loss(input, target, reduction=self.reduction)
