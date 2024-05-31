"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

from collections import OrderedDict  # pylint: disable=g-importing-member

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.hub import download_cached_file

from ..tuning_modules import PadPrompter, ConvAdapter

__all__ = [
  'resnet50_bit_m'
]

class StdConv2d(nn.Conv2d):

  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                   padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias)


def tf2th(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
  """Pre-activation (v2) bottleneck block.
  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, cin, cout=None, cmid=None, stride=1, tuning_config={'method':'full'}):
    super().__init__()
    cout = cout or cin
    cmid = cmid or cout//4

    self.gn1 = nn.GroupNorm(32, cin)
    self.conv1 = conv1x1(cin, cmid, bias=tuning_config['method'] == 'bias')
    self.gn2 = nn.GroupNorm(32, cmid)
    self.conv2 = conv3x3(cmid, cmid, stride, bias=tuning_config['method'] == 'bias')  # Original code has it on conv1!!
    self.gn3 = nn.GroupNorm(32, cmid)
    self.conv3 = conv1x1(cmid, cout, bias=tuning_config['method'] == 'bias')
    self.relu = nn.ReLU(inplace=True)

    if (stride != 1 or cin != cout):
      # Projection also with pre-activation according to paper.
      self.downsample = conv1x1(cin, cout, stride)

    self.tuning_config = tuning_config
    if 'conv_adapt' in self.tuning_config['method']:
        # using ReLU and BatchNorm for resnet
        self.tuning_module = ConvAdapter(cmid, cmid, 
                                         kernel_size=3, 
                                         padding=1,
                                         width=int(cmid//tuning_config['adapt_size']), 
                                         stride=stride, 
                                         groups=int(cmid//tuning_config['adapt_size']), 
                                         dilation=1,
                                         act_layer=nn.ReLU)
    


  def forward(self, x):
    out = self.relu(self.gn1(x))

    # Residual branch
    residual = x
    if hasattr(self, 'downsample'):
      residual = self.downsample(out)

    # Unit's branch
    out = self.conv1(out)
    if 'conv_adapt' in self.tuning_config['method']:
        out_adapt = self.tuning_module(out)
    out = self.conv2(self.relu(self.gn2(out)))
    if 'conv_adapt' in self.tuning_config['method']:
        out = out + out_adapt
    out = self.conv3(self.relu(self.gn3(out)))

    return out + residual

  def load_from(self, weights, prefix=''):
    convname = 'standardized_conv2d'
    with torch.no_grad():
      self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
      self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
      self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
      self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
      self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
      self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
      self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
      self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
      self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
      if hasattr(self, 'downsample'):
        w = weights[f'{prefix}a/proj/{convname}/kernel']
        self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
  """Implementation of Pre-activation (v2) ResNet mode."""

  def __init__(self, block_units, width_factor, head_size=21843, zero_head=False,
                     input_resolution=224, tuning_config={'method':'full'}):
    super().__init__()
    # tuning config
    self.tuning_config = tuning_config
    if self.tuning_config['method'] == 'prompt':
        self.tuning_module = PadPrompter(prompt_size=self.tuning_config['prompt_size'], image_size=input_resolution)

    wf = width_factor  # shortcut 'cause we'll use it a lot.

    # The following will be unreadable if we split lines.
    # pylint: disable=line-too-long
    self.root = nn.Sequential(OrderedDict([
        ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
        ('pad', nn.ConstantPad2d(1, 0)),
        ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        # The following is subtly not the same!
        # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    self.body = nn.Sequential(OrderedDict([
        ('block1', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf, tuning_config=self.tuning_config))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf, tuning_config=self.tuning_config)) for i in range(2, block_units[0] + 1)],
        ))),
        ('block2', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2, tuning_config=self.tuning_config))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf, tuning_config=self.tuning_config)) for i in range(2, block_units[1] + 1)],
        ))),
        ('block3', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2, tuning_config=self.tuning_config))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf, tuning_config=self.tuning_config)) for i in range(2, block_units[2] + 1)],
        ))),
        ('block4', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2, tuning_config=self.tuning_config))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf, tuning_config=self.tuning_config)) for i in range(2, block_units[3] + 1)],
        ))),
    ]))
    # pylint: enable=line-too-long

    self.zero_head = zero_head
    # self.before_head = nn.Sequential(OrderedDict([
    #     ('gn', nn.GroupNorm(32, 2048*wf)),
    #     ('relu', nn.ReLU(inplace=True)),
    #     ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
    #     # ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
    # ]))
    self.norm = nn.GroupNorm(32, 2048*wf)
    # self.relu = nn.ReLU()
    self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

    self.head = nn.Identity()
    self.num_features = 2048*wf


    # initialize bias and weights of conv nets
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.xavier_uniform(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)  



  def forward(self, x):
    # Prompt tuning
    if self.tuning_config['method'] == 'prompt':
          x = self.tuning_module(x)
    x = self.body(self.root(x))
    x = self.norm(x)
    x = F.relu(x)
    x = self.avg_pool(x)
    assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
    return self.head(x[...,0,0])

  def load_from(self, weights, prefix='resnet/'):
    with torch.no_grad():
      self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
      self.norm.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
      self.norm.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
      # if self.zero_head:
      #   nn.init.zeros_(self.before_head.conv.weight)
      #   nn.init.zeros_(self.before_head.conv.bias)
      # else:
      #   self.before_head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
      #   self.before_head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))
      for bname, block in self.body.named_children():
        for uname, unit in block.named_children():
          unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


# KNOWN_MODELS = OrderedDict([
#     ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
#     ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
#     ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
#     ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
#     ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
#     ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
#     ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
#     ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
#     ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
#     ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
#     ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
#     ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
# ])


model_urls = {
    "resnet50_bit_s": "https://storage.googleapis.com/bit_models/BiT-S-R50x1.npz", # IMAGENET1K_V2 Weights, Acc 80.858
    "resnet101_bit_s": "https://storage.googleapis.com/bit_models/BiT-S-R101x1.npz", # Acc 81.886
    "resnet50_bit_m": "https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz", # IMAGENET1K_V2 Weights, Acc 80.858
    "resnet101_bit_m": "https://storage.googleapis.com/bit_models/BiT-M-R101x1.npz", # Acc 81.886
}


def resnet50_bit_s(pretrained=False, **kwargs):
    model = ResNetV2([3, 4, 6, 3], 1, **kwargs)
    if pretrained:
        model.load_from(np.load(download_cached_file(model_urls['resnet50_bit_s'])))
    return model


def resnet101_bit_s(pretrained=False, **kwargs):
    model = ResNetV2([3, 4, 6, 3], 3, **kwargs)
    if pretrained:
        model.load_from(np.load(download_cached_file(model_urls['resnet101_bit_s'])))
    return model
  


def resnet50_bit_m(pretrained=False, **kwargs):
    model = ResNetV2([3, 4, 6, 3], 1, **kwargs)
    if pretrained:
        model.load_from(np.load(download_cached_file(model_urls['resnet50_bit_m'])))
    return model


def resnet101_bit_m(pretrained=False, **kwargs):
    model = ResNetV2([3, 4, 6, 3], 3, **kwargs)
    if pretrained:
        model.load_from(np.load(download_cached_file(model_urls['resnet101_bit_m'])))
    return model
  

if __name__ == '__main__':
  model = resnet50_bit_m(pretrained=True,  tuning_config={'method':'full', 'adapt_size':4, 'adapt_scale':1.0})
  print(model)

  n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  # print("Model = %s" % str(model_without_ddp))
  print('Number of trainable params:', n_parameters)


  x = torch.randn((1, 3, 224, 224))
  o = model(x)
  print(o.shape)
  # for name, param in model.named_parameters():
  #     if name.startswith('head'):
  #         continue
      
  #     if 'tuning_module' in name:
  #         continue

  #     if 'bn' in name:
  #         continue

  #     if 'gn3' in name:
  #         continue

  #     if 'norm' in name:
  #         continue

  #     param.requires_grad = False    
  
  # for name, param in model.named_parameters():
  #   if param.requires_grad == True:
  #       print(name)