import torch
from torch import nn


class AttentionMLP(nn.Module):
    """An improved MLP with two attention layers."""

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden_layers, activation):
        super(AttentionMLP, self).__init__()

        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(input_dim, hidden_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x, return_hidden=False):

        u = self.activation(self.encoder1(x))
        v = self.activation(self.encoder2(x))

        hs = [u, v]

        x = self.activation(self.input_layer(x))
        x = (1 - x) * u + x * v 
        hs.append(x)

        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = (1 - x) * u + x * v
            hs.append(x)
        
        x = self.output_layer(x)

        if return_hidden:
            return x, hs
        else:
            return x
        