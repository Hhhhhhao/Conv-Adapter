from collections import OrderedDict
from typing import Tuple, Union
import hashlib
import os
import urllib
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm



from ..tuning_modules import PadPrompter, ConvAdapter, LinearAdapter

__all__ = [
    "resnet50_clip",
    "resnet101_clip",
    "resnet50x4_clip",
    "resnet50x16_clip",
    "resnet50x64_clip",
]


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, tuning_config=None):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=tuning_config['method'] == 'bias')
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=tuning_config['method'] == 'bias')
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=tuning_config['method'] == 'bias')
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=True)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

        self.tuning_config = tuning_config
        if 'conv_adapt' in self.tuning_config['method']:
            # using ReLU and BatchNorm for resnet
            self.tuning_module = ConvAdapter(planes, planes, 
                                             kernel_size=3, 
                                             padding=1,
                                             width=int(planes//tuning_config['adapt_size']), 
                                             stride=1, 
                                             groups=int(planes//tuning_config['adapt_size']), 
                                             dilation=1,
                                             act_layer=nn.ReLU)

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))

        if 'conv_adapt' in self.tuning_config['method']:
            out_adapt = self.tuning_module(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        if 'conv_adapt' in self.tuning_config['method']:
            out = out + out_adapt
        out = self.avgpool(out)
        
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, tuning_config: dict={'method':'full'}):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        self.tuning_config = tuning_config
        if 'conv_adapt' in self.tuning_config['method']:
            self.tuning_module = LinearAdapter(embed_dim, output_dim, int(embed_dim // self.tuning_config['adapt_size']), act_layer=nn.ReLU)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        if 'conv_adapt' in self.tuning_config['method']:
            # x = torch.cat([x[:1], self.tuning_module.expand(-1, x.shape[1], -1), x[1:]])
            x_adapt = self.tuning_module(x)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        if 'conv_adapt' in self.tuning_config['method']:
            # x = torch.cat([x[:1], self.tuning_module.expand(-1, x.shape[1], -1), x[1:]])
            x = x + x_adapt
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, 
                 tuning_config=None):
        super().__init__()
        # tuning config
        self.tuning_config = tuning_config
        self.tuning_module = None
        if self.tuning_config['method'] == 'prompt':
            self.tuning_module = PadPrompter(prompt_size=self.tuning_config['prompt_size'], image_size=input_resolution)

        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, tuning_config=self.tuning_config)
        self.num_features  = output_dim
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.num_features = embed_dim

        self.head = nn.Identity()

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


    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, tuning_config=self.tuning_config)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, tuning_config=self.tuning_config))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        # x = x.type(self.conv1.weight.dtype)

        # Prompt tuning
        if self.tuning_config['method'] == 'prompt':
            x = self.tuning_module(x)

        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        # x = self.pool(x).reshape(-1, self.num_features)
        
        x = self.head(x)
        return x


model_urls = {
    "resnet50_clip": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "resnet101_clip": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "resnet50x4_clip": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "resnet50x16_clip": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "resnet50x64_clip": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
}



def _load_checkpoint(model_name):
    model_path = _download(model_urls[model_name], os.path.expanduser("~/.cache/clip"))

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu") 
    return state_dict or model.state_dict()


def _resize_pos_embed(posemb, posemb_new):
    import math
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    posemb_tok, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1
    posemb_new_grid = posemb_new[1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(len(posemb_new_grid)))
    print('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=0)
    return posemb


def _load_model(state_dict, input_resolution, **kwargs):
    counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    pretrain_image_resolution = output_width * 32
    embed_dim = state_dict["text_projection"].shape[1]
        
    vision_heads = vision_width * 32 // 64
    model = ModifiedResNet(layers=vision_layers,
                           output_dim=embed_dim,
                           heads=vision_heads,
                           input_resolution=input_resolution,
                           width=vision_width, 
                           **kwargs)

    # TODO: deal with input_size for x4 models
    if pretrain_image_resolution != input_resolution:
        # interpolate the position encoding
        state_dict["visual.attnpool.positional_embedding"] = _resize_pos_embed(state_dict["visual.attnpool.positional_embedding"], model.attnpool.positional_embedding)

    return model
    

def _adapt_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('visual'):
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = value
    return new_state_dict


def resnet50_clip(pretrained=False, input_resolution=224, **kwargs):
    state_dict = _load_checkpoint("resnet50_clip")
    model = _load_model(state_dict, input_resolution, **kwargs)
    if pretrained:
        state_dict = _adapt_state_dict(state_dict)
        keys = model.load_state_dict(state_dict, strict=False)
         
    return model


def resnet101_clip(pretrained=False, input_resolution=224, **kwargs):
    state_dict = _load_checkpoint("resnet101_clip")
    model = _load_model(state_dict, input_resolution, **kwargs)
    if pretrained:
        state_dict = _adapt_state_dict(state_dict)
        keys = model.load_state_dict(state_dict, strict=False)
         
    return model


def resnet50x4_clip(pretrained=False, input_resolution=224, **kwargs):
    state_dict = _load_checkpoint("resnet50x4_clip")
    model = _load_model(state_dict, input_resolution, **kwargs)
    if pretrained:
        state_dict = _adapt_state_dict(state_dict)
        keys = model.load_state_dict(state_dict, strict=False)
         
    return model


def resnet50x16_clip(pretrained=False, input_resolution=224, **kwargs):
    state_dict = _load_checkpoint("resnet50x16_clip")
    model = _load_model(state_dict, input_resolution, **kwargs)
    if pretrained:
        state_dict = _adapt_state_dict(state_dict)
        keys = model.load_state_dict(state_dict, strict=False)
         
    return model


def resnet50x64_clip(pretrained=False, input_resolution=224, **kwargs):
    state_dict = _load_checkpoint("resnet50x64_clip")
    model = _load_model(state_dict, input_resolution, **kwargs)
    if pretrained:
        state_dict = _adapt_state_dict(state_dict)
        keys = model.load_state_dict(state_dict, strict=False)
         
    return model


if __name__ == '__main__':
    model = resnet50x4_clip(pretrained=True, tuning_config={'method':'full', 'adapt_size': 4})
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model_without_ddp))
    print('Number of trainable params:', n_parameters)

    x = torch.randn((2, 3, 224, 224))
    o = model(x)
