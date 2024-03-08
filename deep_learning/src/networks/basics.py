import torch
from torch import nn


class LambdaLayer(nn.Module):
    """Apply a user-defined lambda function."""

    def __init__(self, lambd, name=""):
        super().__init__()

        if not callable(lambd):
            raise TypeError(f"Argument lambd should be callable, got {repr(type(lambd).__name__)}")
        self.lambd = lambd
        self.name = name

    def forward(self, x):
        return self.lambd(x)
    
    def extra_repr(self):
        return self.name


class Scaler(nn.Module):
    """Scale input (v, x) by constants."""

    def __init__(self, scaling_factors=(1., 1.)):
        super(Scaler, self).__init__()

        self.scaling_factors = scaling_factors
        self.register_buffer("v_factor", torch.tensor(scaling_factors[0]))
        self.register_buffer("x_factor", torch.tensor(scaling_factors[1]))

    def forward(self, u):
        v, x = u.chunk(2, dim=-1)
        return torch.cat((self.v_factor * v, self.x_factor * x), dim=-1)

    def extra_repr(self):
        return f"scaling_factors={self.scaling_factors}"


class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(self, layer_sizes, activation, use_bn=False):
        super(MLP, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-1)]
        )
        self.activation = activation
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-2)]
            )
        
    def forward(self, x, return_hidden=False):
        hs = []
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            if return_hidden:
                hs.append(x)
        x = self.layers[-1](x)

        if return_hidden:
            return x, hs 
        else:         
            return x


class FrictionBlock(nn.Module):
    """Friction Block."""
    
    def __init__(self, init_gamma=0.):
        super(FrictionBlock, self).__init__()

        self.init_gamma = init_gamma
        self.gamma = nn.Parameter(torch.tensor(init_gamma), requires_grad=True)
        
    def forward(self, x):
        p, q = x.chunk(2, dim=-1)
        dp = - self.gamma**2 * p
        dq = torch.zeros_like(q)
        return torch.cat((dp, dq), dim=-1)

    def extra_repr(self):
        return f"init_gamma={self.init_gamma}"