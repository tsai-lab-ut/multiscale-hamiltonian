"""
Tiramisu network (FC-DenseNet) where convolution layers are replaced by linear layers.

Reference: https://github.com/bfortuner/pytorch_tiramisu
"""

import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """Dense block."""
    
    def __init__(self, in_features, growth_rate, n_layers, activation, use_bn):
        super(DenseBlock, self).__init__()

        self.linears = nn.ModuleList(
            [nn.Linear(in_features + i*growth_rate, growth_rate) for i in range(n_layers)]
        )
        self.activation = activation
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(growth_rate) for _ in range(n_layers)]
            )

    def forward(self, x):
        new_features = []

        for i in range(len(self.linears)):
            out = self.linears[i](x)
            if self.use_bn:
                out = self.bn_layers[i](out)
            out = self.activation(out)
            x = torch.cat([x, out], -1)  # -1 = feature axis
            new_features.append(out)
        
        return torch.cat(new_features, -1)


class TiramisuNet(nn.Module):
    """Tiramisu network."""

    def __init__(self, in_features, activation, expand_blocks=(5,5,5,5,5),
                 shrink_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, use_bn=False):
        super(TiramisuNet, self).__init__()

        cur_features_count = in_features
        skip_connection_feature_counts = [in_features]

        ##################
        # Expanding path #
        ##################

        self.dense_blocks_expand = nn.ModuleList([])
        for i in range(len(expand_blocks)):
            self.dense_blocks_expand.append(
                DenseBlock(cur_features_count, growth_rate, expand_blocks[i], 
                           activation, use_bn))
            cur_features_count += (growth_rate*expand_blocks[i])
            skip_connection_feature_counts.insert(0, cur_features_count)

        #####################
        #     Bottleneck    #
        #####################

        self.bottleneck = DenseBlock(cur_features_count, growth_rate, bottleneck_layers, 
                                     activation, use_bn)
        prev_block_features = growth_rate*bottleneck_layers

        #######################
        #   Shrinking path   #
        #######################

        self.dense_blocks_shrink = nn.ModuleList([])
        for i in range(len(shrink_blocks)):
            cur_features_count = prev_block_features + skip_connection_feature_counts[i]
            self.dense_blocks_shrink.append(
                DenseBlock(cur_features_count, growth_rate, shrink_blocks[i],
                           activation, use_bn))
            prev_block_features = growth_rate*shrink_blocks[i]

        ## Final layer ##

        cur_features_count = prev_block_features + skip_connection_feature_counts[-1]
        self.final_layer = nn.Linear(cur_features_count, in_features)

    def forward(self, x):

        skip_connections = [x]
        for dense_block in self.dense_blocks_expand:
            out = dense_block(x)
            x = torch.cat([x, out], -1)
            skip_connections.append(x)

        x = self.bottleneck(x)
        for dense_block in self.dense_blocks_shrink:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], -1)
            x = dense_block(x)

        skip = skip_connections.pop()
        x = torch.cat([x, skip], -1)
        x = self.final_layer(x)
        
        return x
