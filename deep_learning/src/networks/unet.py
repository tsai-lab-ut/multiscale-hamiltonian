import math
import torch
from torch import nn


def get_conv_layer(in_channels, out_channels, kernel_size):
    """
    Define a stride 1 convolution layer which keeps the input image size unchanged.
    (kernel_size is required to be an odd number because Conv2d allows only symmetric padding) 
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2) 


def get_downscale_layer(in_channels, out_channels, kernel_size):
    """Define a stride 2 convolution layer which halves the input image size."""
    return nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=math.ceil(kernel_size/2-1))


def get_batchnorm_layer(num_features):
    """Define a batch normalization layer."""
    return nn.BatchNorm1d(num_features)


class Concatenate(nn.Module):
    """Concatenate two inputs along the channel dimension."""
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return x


class Up(nn.Module):
    """Upscaling with [Convolution, Batch Norm, Activation] * n, 2x Upsampling, Batch Norm, Activation."""

    def __init__(self, in_channels, out_channels, activation, kernel_size=3, n_conv=1, use_bn=True, upscale=True):
        super(Up, self).__init__()

        self.n_conv = n_conv
        self.use_bn = use_bn
        self.upscale = upscale
        self.activation = activation

        # conv layers 
        self.conv_layers = nn.ModuleList(
            [get_conv_layer(in_channels, out_channels, kernel_size)] + \
            [get_conv_layer(out_channels, out_channels, kernel_size) for _ in range(self.n_conv-1)]
        )

        # upsampling layer 
        if self.upscale:
            self.up_layer = nn.Upsample(scale_factor=2, mode="nearest")

        # batch norm layers 
        if self.use_bn:
            self.bn_layers = nn.ModuleList([get_batchnorm_layer(out_channels) for _ in range(self.n_conv)])
            if self.upscale:
                self.up_bn = get_batchnorm_layer(out_channels)

    def forward(self, x):

        for i in range(self.n_conv):
            x = self.conv_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)

        before_upscale = x
        if self.upscale:
            x = self.up_layer(x)
            if self.use_bn:
                x = self.up_bn(x)
            x = self.activation(x)

        return x, before_upscale


class Down(nn.Module):
    """Downscaling with [Stride 2 Convolution, Batch Norm, Activation], Merge, [Convolution, Batch Norm, Activation] * n."""

    def __init__(self, in_channels, out_channels, activation, kernel_size=3, n_conv=1, use_bn=True):
        super(Down, self).__init__()

        self.n_conv = n_conv
        self.use_bn = use_bn
        self.activation = activation

        # downscaling layer 
        self.down_layer = get_downscale_layer(in_channels, out_channels, kernel_size)

        # concatenate layer
        self.concat = Concatenate()

        # conv layers 
        self.conv_layers = nn.ModuleList(
            [get_conv_layer(in_channels, out_channels, kernel_size)] + \
            [get_conv_layer(out_channels, out_channels, kernel_size) for _ in range(self.n_conv-1)]
        )

        # batch norm layers 
        if self.use_bn:
            self.bn0 = get_batchnorm_layer(out_channels)
            self.bn_layers = nn.ModuleList([get_batchnorm_layer(out_channels) for _ in range(self.n_conv)])        

    def forward(self, x, xskip):

        x = self.down_layer(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.activation(x)

        x = self.concat(x, xskip)

        for i in range(self.n_conv):
            x = self.conv_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)

        return x

    
class UNet1D(nn.Module):
    """1D U-Net."""

    def __init__(self, d, in_channels, out_channels, activation, n_blocks=3, k=4, kernel_size=3, use_bn=True, n_conv=1, 
                 fc_n_nodes=500, fc_n_layers=2):
        super(UNet1D, self).__init__()

        self.d = d
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.n_blocks = n_blocks 
        self.k = k 
        self.kernel_size = kernel_size 
        self.use_bn = use_bn
        self.n_conv = n_conv
        self.fc_n_nodes = fc_n_nodes
        self.fc_n_layers = fc_n_layers
        self.activation = activation 

        up_blocks = []
        down_blocks = []
         
        for i in range(self.n_blocks+1):
            n_in = self.in_channels if i == 0 else n_out
            n_out = 2**i * self.k
            upscale = False if i == self.n_blocks else True 
            up = Up(
                in_channels=n_in, 
                out_channels=n_out, 
                kernel_size=self.kernel_size,
                n_conv=self.n_conv,
                use_bn=self.use_bn,
                upscale=upscale,
                activation=self.activation
            )
            up_blocks.append(up)

        for i in range(self.n_blocks):
            n_in = n_out
            n_out = n_in // 2
            down = Down(
                in_channels=n_in, 
                out_channels=n_out, 
                kernel_size=self.kernel_size,
                n_conv=self.n_conv,
                use_bn=self.use_bn,
                activation=self.activation
            )
            down_blocks.append(down)

        self.up_blocks = nn.ModuleList(up_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)

        n_channels = 2**self.n_blocks * self.k 
        width = 2**self.n_blocks * self.d 
        fc_layers = [nn.Flatten()]
        for i in range(self.fc_n_layers):
            n_in = n_channels*width if i == 0 else self.fc_n_nodes
            n_out = self.fc_n_nodes 
            fc_layers.append(nn.Linear(n_in, n_out))
            fc_layers.append(self.activation)
        fc_layers.append(nn.Linear(self.fc_n_nodes, n_channels*width))
        fc_layers.append(nn.Unflatten(1, (n_channels, width)))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.final_conv = get_conv_layer(in_channels=self.k, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x, return_hidden=False):

        x = torch.reshape(x, (-1, 2, self.d))

        xs = []
        for i, up in enumerate(self.up_blocks):
            x, before_upscale = up(x)
            xs.append(before_upscale)

        x = self.fc_layers(x)

        if return_hidden:
            hidden = x
    
        for i, down in enumerate(self.down_blocks):
            x = down(x, xs[-2-i])

        x = self.final_conv(x) 

        x = torch.reshape(x, (-1, 2*self.d))

        if return_hidden:
            return x, hidden
        else:
            return x
