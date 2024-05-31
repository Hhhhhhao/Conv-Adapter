from sqlite3 import adapters
from typing import Callable, List, Optional
import torch
from torch import Tensor
import torch.nn as nn

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        out_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, out_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        # return scale * input
        return scale


class ConvAdapterDesign1(nn.Module):
    """
    Design 1
    """
    def __init__(self, inplanes, outplanes, width, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, norm_layer=None, act_layer=None, adapt_scale=1.0):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if act_layer is None:
            act_layer = nn.Identity

        # point-wise conv
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1)
        self.norm1 = norm_layer(width)

        # depth-wise conv
        self.conv2 = nn.Conv2d(width, width, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=int(dilation))
        self.norm2 = norm_layer(width)

        # poise-wise conv
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1)
        self.norm3 = norm_layer(outplanes)

        self.act = act_layer()

        self.adapt_scale = adapt_scale
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.norm2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.act(out)

        return out * self.adapt_scale

class ConvAdapter(nn.Module):
    """
    Design 2 v4
    """
    def __init__(self, inplanes, outplanes, width, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, norm_layer=None, act_layer=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if act_layer is None:
            act_layer = nn.Identity

        # self.act = nn.SiLU()

        # depth-wise conv
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=int(dilation))
        # self.norm = norm_layer(width)
        self.act = act_layer()

        # poise-wise conv
        self.conv2 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1)

        # se 
        # self.se = SqueezeExcitation(inplanes, width, outplanes, activation=act_layer)
        self.se = nn.Parameter(1.0 * torch.ones((1, outplanes, 1, 1)), requires_grad=True)
        # self.se = 4.0

    
    def forward(self, x):
        out = self.conv1(x)
        # out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)
        out = out * self.se
        # TODO: add norm layer

        return out
    


class LinearAdapter(nn.Module):
    """
    Design 2 v4
    """
    def __init__(self, inplanes, outplanes, width, act_layer=None, **kwargs):
        super().__init__()

        self.fc1 = nn.Linear(inplanes, width)
        self.fc2 = nn.Linear(width, outplanes)
        self.act = act_layer()
        self.se = nn.Parameter(1.0 * torch.ones((1, outplanes)), requires_grad=True)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.norm(out)
        out = self.act(out)
        out = self.fc2(out)
        out = out * self.se
        return out


if __name__ == '__main__':
    adapter = ConvAdapter(128, 128, width=32, groups=32)
    print(adapter.conv1.weight.shape)
    print(adapter.conv2.weight.shape)