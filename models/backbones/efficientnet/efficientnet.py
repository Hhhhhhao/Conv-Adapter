
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from .efficientnet_blocks import SqueezeExcite
from .efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights, round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from timm.models.layers import create_conv2d, create_classifier, get_norm_act_layer, GroupNormAct

__all__ = [
    'tf_efficientnetv2_l_in21k',
    'tf_efficientnetv2_m_in21k',
]

class EfficientNet(nn.Module):
    """ EfficientNet
    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large, XL & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * MobileNet-V2
      * FBNet C
      * Single-Path NAS Pixel1
      * TinyNet
    """

    def __init__(
            self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, fix_stem=False,
            output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None,
            se_layer=None, drop_rate=0., drop_path_rate=0., global_pool='avg',
            input_resolution=224,
            tuning_config={'method':'full'}):
        super(EfficientNet, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.input_resoltuion = input_resolution

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate, tuning_config=tuning_config)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_act_layer(self.num_features, inplace=True)
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)
        del self.classifier
        self.head = nn.Identity()
        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'conv_head|bn2', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.blocks(x)

        x = self.conv_head(x)
        x = self.bn2(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x, pre_logits=True)
        x = self.head(x)
        return x

model_urls = {
    "tf_efficientnetv2_m_in21k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m_21k-361418a2.pth",
    "tf_efficientnetv2_l_in21k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21k-91a19ec9.pth",
}


def tf_efficientnetv2_m_in21k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Medium w/ ImageNet-21k pretrained weights. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'

    arch_def = [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, 1.0),
        num_features=1280,
        stem_size=24,
        round_chs_fn=partial(round_channels, multiplier=1.0),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = EfficientNet(**model_kwargs)
    if pretrained:
        url = model_urls["tf_efficientnetv2_m_in21k"] 
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        keys = model.load_state_dict(checkpoint, strict=False)
    return model


def tf_efficientnetv2_l_in21k(pretrained=False, **kwargs):
    """ EfficientNet-V2 Large w/ ImageNet-21k pretrained weights. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, 1.0),
        num_features=1280,
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=1.0),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'silu'),
        **kwargs,
    )
    model = EfficientNet(**model_kwargs)
    if pretrained:
        url = model_urls["tf_efficientnetv2_l_in21k"] 
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        keys = model.load_state_dict(checkpoint, strict=False)
    return model


if __name__ == '__main__':
    model = tf_efficientnetv2_l_in21k(pretrained=True, tuning_config={'method':'full', 'adapt_size':2})
    x = torch.randn((2, 3, 384, 384))
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
    #     if param.requires_grad == True:
    #         print(name)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)