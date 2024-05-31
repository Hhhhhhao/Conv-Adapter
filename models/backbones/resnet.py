from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..tuning_modules import PadPrompter, ConvAdapter, LinearAdapter, ProgramModule

__all__ = [
    'resnet50',
    'resnet50_mocov3',
    'resnet101',
    'resnet152',
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool=False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool=False) -> nn.Conv2d:
    """1x1 convolution"""
    # TODO: change back to False
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        tuning_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, bias=tuning_config['method'] == 'bias')
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=tuning_config['method'] == 'bias')
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.tuning_config = tuning_config
        if 'conv_adapt' in self.tuning_config['method']:
            # using ReLU and BatchNorm for resnet
            self.tuning_module = ConvAdapter(inplanes, planes, 
                                             kernel_size=3, 
                                             padding=1,
                                             width=inplanes // tuning_config['adapt_size'], 
                                             stride=stride, 
                                             groups=inplanes // tuning_config['adapt_size'], 
                                             dilation=1,
                                             act_layer=nn.ReLU)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        if 'conv_adapt' in self.tuning_config['method']:
            out_adapt = self.tuning_module(out)
        out = self.conv1(x)
        if 'conv_adapt' in self.tuning_config['method']:
            out = out + out_adapt
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        tuning_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, bias=tuning_config['method'] == 'bias')
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, bias=tuning_config['method'] == 'bias')
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=tuning_config['method'] == 'bias')
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.tuning_config = tuning_config
        if 'conv_adapt' in self.tuning_config['method']:
            # self.tuning_module1 = LinearAdapter(inplanes, width, width=max(1, inplanes//tuning_config['adapt_size']//4), act_layer=nn.ReLU)
            self.tuning_module = ConvAdapter(width, width, 
                                             kernel_size=3, 
                                             padding=1,
                                             width=int(width//tuning_config['adapt_size']), 
                                             stride=stride, 
                                             groups=int(width//tuning_config['adapt_size']), 
                                             dilation=1,
                                             act_layer=nn.ReLU)
            # self.tuning_module3 = LinearAdapter(width, planes * self.expansion, width=max(1, width//tuning_config['adapt_size']), act_layer=nn.ReLU)
            # self.tuning_module = conv1x1(width, width)


    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # if 'conv_adapt' in self.tuning_config['method']:
        #     out_adapt = self.tuning_module(out)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if 'conv_adapt' in self.tuning_config['method']:
            out_adapt = self.tuning_module(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if 'conv_adapt' in self.tuning_config['method']:
            out = out + out_adapt

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # if 'conv_adapt' in self.tuning_config['method']:
        #    out = out + out_adapt
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_resolution: Optional[int] = 224,
        tuning_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        # tuning config
        self.tuning_config = tuning_config
        if self.tuning_config['method'] == 'prompt':
            self.tuning_module = PadPrompter(prompt_size=self.tuning_config['prompt_size'], image_size=input_resolution)

        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512 * block.expansion, num_classes)
        self.num_features = 512 * block.expansion


        # initialize bias and weights of conv nets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
        self.head = nn.Identity()
        self.norm = nn.BatchNorm2d(512 * block.expansion)


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, tuning_config=self.tuning_config
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    tuning_config=self.tuning_config
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        
        # Prompt tuning
        if self.tuning_config['method'] == 'prompt':
            x = self.tuning_module(x)
        
        if 'repnet' in self.tuning_config['method']:
            side_x = self.tuning_module_pool(x) 
            side_x = self.tuning_module[0](side_x) 

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = x + side_x

            for i in range(len(self.layer1)):
                if i == 0:
                    main_x = self.layer1[i](x)
                    side_x = self.tuning_module[1](x)
                    x = main_x + side_x
                    x+=side_x
                else:
                    x = self.layer1[i](x)
                # print(x.shape, side_x.shape)

            for i in range(len(self.layer2)):
                if i == 0:
                    main_x = self.layer2[i](x)
                    side_x = self.tuning_module[2](x)
                    x = main_x + side_x
                else:
                    x = self.layer2[i](x)
                # print(x.shape)
            for i in range(len(self.layer3)):
                if i == 0:
                    main_x = self.layer3[i](x)
                    side_x = self.tuning_module[3](x)
                    x = main_x + side_x
                else:                
                    x = self.layer3[i](x)
                # print(x.shape)
            for i in range(len(self.layer4)):
                if i == 0:
                    main_x = self.layer4[i](x)
                    side_x = self.tuning_module[4](x)
                    x = main_x + side_x
                elif i == 2:
                    main_x = self.layer4[i](x)
                    side_x = self.tuning_module[5](x)
                    x = main_x + side_x                
                else:  
                    x = self.layer4[i](x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


model_urls = {
    "resnet50_1k": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth", # IMAGENET1K_V2 Weights, Acc 80.858
    "resnet101_1k": "https://download.pytorch.org/models/resnet101-cd907fc2.pth", # Acc 81.886
    "resnet152_1k": "https://download.pytorch.org/models/resnet152-f82ba261.pth", # Acc  82.284
    "resnet50_mocov3": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar"
}


def resnet50(pretrained=False, in_22k=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        url = model_urls["resnet50_22k"] if in_22k else model_urls["resnet50_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        keys = model.load_state_dict(checkpoint, strict=False)
         
    return model

def resnet50_mocov3(pretrained=False, in_22k=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        url = model_urls["resnet50_mocov3"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.base_encoder'):
                new_key = '.'.join(key.split('.')[2:])
                new_state_dict[new_key] = value
        keys = model.load_state_dict(new_state_dict, strict=False)
    return model


def resnet101(pretrained=False, in_22k=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        url = model_urls["resnet101_22k"] if in_22k else model_urls["resnet101_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        keys = model.load_state_dict(checkpoint, strict=False)
         
    return model


def resnet152(pretrained=False, in_22k=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        url = model_urls["resnet152_22k"] if in_22k else model_urls["resnet152_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        keys = model.load_state_dict(checkpoint, strict=False)
         
    return model


if __name__ == '__main__':
    model = resnet50(pretrained=True, tuning_config={'method':'full', 'adapt_size': 16})
    print(model)
    x = torch.randn((1, 3, 224, 224))
    o = model(x)

    # for name, param in model.named_parameters():
    #     print(name)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('Number of trainable params:', n_parameters)