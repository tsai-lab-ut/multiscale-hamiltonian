import torch
from torch import nn


class HamiltonianReversibleBlock(nn.Module):
    """Hamiltonian Reversible Block."""
    
    def __init__(self, degree_of_freedom, activation):
        super(HamiltonianReversibleBlock, self).__init__()
        
        self.dof = degree_of_freedom
        self.activation = activation
        
        self.layer1 = nn.Linear(self.dof, self.dof)
        self.layer2 = nn.Linear(self.dof, self.dof)
        self.h = 1e-3
        
    def forward(self, x):
        
        p, q = torch.split(x, [self.dof, self.dof], dim=-1)
        
        pnew = p + self.h * torch.matmul(self.activation(self.layer1(q)), self.layer1.weight)
        qnew = q - self.h * torch.matmul(self.activation(self.layer2(pnew)), self.layer2.weight)
        
        xnew = torch.cat((pnew, qnew), dim=-1)
        
        return xnew


class HamiltonianReversibleNetwork(nn.Module):
    """Hamiltonian Reversible Network."""
    
    def __init__(self, layer_sizes, activation, use_bn=False):
        super(HamiltonianReversibleNetwork, self).__init__()
        
        self.layer_sizes = layer_sizes
        layers = []
        for i in range(len(self.layer_sizes)-1):
            in_features = self.layer_sizes[i]
            out_features = self.layer_sizes[i+1]
            if in_features == out_features and in_features % 2 == 0:
                layers.append(HamiltonianReversibleBlock(in_features//2, activation))
            else:
                layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)
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
            if not isinstance(self.layers[i], HamiltonianReversibleBlock):
                x = self.activation(x)
            if return_hidden:
                hs.append(x)
        x = self.layers[-1](x)
        if return_hidden:
            return x, hs
        else:
            return x
    
    def compute_weight_smoothness(self):
        s = 0.
        for layer_cur, layer_next in zip(self.layers[:-1], self.layers[1:]):
            if isinstance(layer_cur, HamiltonianReversibleBlock):
                if isinstance(layer_next, HamiltonianReversibleBlock):
                    if layer_cur.dof == layer_next.dof:
                        s += torch.linalg.norm(layer_next.layer1.weight - layer_cur.layer1.weight) 
                        s += torch.linalg.norm(layer_next.layer2.weight - layer_cur.layer2.weight) 
        return s / 1e-3
