import torch
from torch import nn
from networks.basics import MLP


class EquivarianceLayer(nn.Module):
    """Equivariance Layer."""

    def __init__(self, N, n_hidden_nodes, n_hidden_layers, activation, use_bn):
        super(EquivarianceLayer, self).__init__()

        n_in = (2*N)**2
        n_out = 2*N
        self.mlp = MLP([n_in]+[n_hidden_nodes]*n_hidden_layers+[n_out], activation, use_bn)

    def forward(self, x):
        scalars, basis = x
        coeffs = self.mlp(scalars)  # batch_size x 2N
        output = torch.bmm(coeffs.unsqueeze(1), basis).squeeze(1)  # batch_size x d
        return output


class EquivarianceNetwork(nn.Module):
    """Equivariance Network."""

    def __init__(self, N, d, n_hidden_nodes, n_hidden_layers, activation, use_bn):
        super(EquivarianceNetwork, self).__init__()

        self.N = N
        self.d = d
        self.layers = nn.ModuleList(
            EquivarianceLayer(N, n_hidden_nodes, n_hidden_layers, activation, use_bn) for _ in range(2*N)
        )
        
    def forward(self, x, return_hidden=False):
        # x has shape (batch_size x 2Nd)
        x = x.reshape(-1, 2*self.N, self.d)  # batch_size x 2N x d
        scalars = torch.bmm(x, torch.transpose(x, 1, 2))  # batch_size x 2N x 2N
        scalars = scalars.flatten(start_dim=1)  # batch_size x (2N)^2
        output = torch.cat([l((scalars, x)) for l in self.layers], 1)  # batch_size x 2Nd
        if return_hidden:
            return output, scalars
        else:
            return output
            