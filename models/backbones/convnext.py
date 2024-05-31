import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from ..tuning_modules import PadPrompter, ConvAdapter

__all__ = [
    'convnext_tiny',
    'convnext_tiny_22k',
    'convnext_small',
    'convnext_small_22k',
    'convnext_base',
    'convnext_base_22k',
    'convnext_large',
    'convnext_large_22k',
    'convnext_xlarge_22k'
]

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, tuning_config=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.tuning_config = tuning_config
        if 'conv_adapt' in self.tuning_config['method']:
            # using ReLU and BatchNorm for resnet
            self.tuning_module = ConvAdapter(dim, dim, 
                                             kernel_size=7, 
                                             padding=3,
                                             width=int(dim//tuning_config['adapt_size']), 
                                             stride=1, 
                                             groups=int(dim//tuning_config['adapt_size']),
                                             dilation=1,
                                             act_layer=nn.GELU)

    def forward(self, x):
        input = x

        if 'conv_adapt' in self.tuning_config['method']:
            x_adapt = self.tuning_module(x)
        x = self.dwconv(x)
        if 'conv_adapt' in self.tuning_config['method']:
            x = x + x_adapt
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 input_resolution=224,
                 tuning_config=None,
                 ):
        super().__init__()
        # tuning config
        self.tuning_config = tuning_config
        self.tuning_module = None
        if self.tuning_config['method'] == 'prompt':
            self.tuning_module = PadPrompter(prompt_size=self.tuning_config['prompt_size'], image_size=input_resolution)

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.ModuleList([
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),]
            )
            self.downsample_layers.append(downsample_layer)

        if 'conv_adapt' in self.tuning_config['method']:
            self.tuning_module_downsample = nn.ModuleList()
            for i in range(3):
                self.tuning_module_downsample.append(ConvAdapter(dims[i], dims[i+1], 
                                                     kernel_size=2, 
                                                     padding=0,
                                                     width=int(dims[i]//tuning_config['adapt_size']), 
                                                     stride=2, 
                                                     groups=int(dims[i]//tuning_config['adapt_size']),
                                                     dilation=1,
                                                     act_layer=nn.GELU))


        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, tuning_config=self.tuning_config) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.num_features = dims[-1]

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        # delete head
        self.head = nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            for downsample_module in self.downsample_layers[i]:
                if isinstance(downsample_module, nn.Conv2d) and ('conv_adapt' in self.tuning_config['method']) and i >0:
                    x_adapt = self.tuning_module_downsample[i-1](x)
                x = downsample_module(x)
                if isinstance(downsample_module, nn.Conv2d) and ('conv_adapt' in self.tuning_config['method'])  and i > 0:
                    x = x + x_adapt
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        if self.tuning_config['method'] == 'prompt':
            x = self.tuning_module(x)
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

def convnext_tiny(pretrained=False,**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        keys = model.load_state_dict(checkpoint["model"], strict=False)
         
    return model

def convnext_tiny_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        keys = model.load_state_dict(checkpoint["model"], strict=False)
         
    return model

def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        keys = model.load_state_dict(checkpoint["model"], strict=False)
         
    return model

def convnext_small_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        keys = model.load_state_dict(checkpoint["model"], strict=False)
         
    return model


def convnext_base(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        keys = model.load_state_dict(checkpoint["model"], strict=False)
         
    return model

def convnext_base_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        keys = model.load_state_dict(checkpoint["model"], strict=False)
         
    return model


def convnext_large(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        keys = model.load_state_dict(checkpoint["model"], strict=False)
         
    return model


def convnext_large_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        keys = model.load_state_dict(checkpoint["model"], strict=False)
         
    return model


def convnext_xlarge_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        keys = model.load_state_dict(checkpoint["model"], strict=False)
        #  
    return model



if __name__ == '__main__':
  model = convnext_large(pretrained=True,  tuning_config={'method':'full', 'adapt_size':4, 'adapt_scale':1.0})
  # print(model)
  print(model)

  n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model_without_ddp))
  print('Number of trainable params:', n_parameters)
  x = torch.randn((1, 3, 224, 224))
  o = model(x)
  print(o.shape)