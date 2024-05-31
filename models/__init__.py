import torch
import torch.nn as nn


from .backbones import *
from .heads import *
from .tuning_modules import set_tuning_config
from .layers.ws_conv import WSConv2d

__all__ = ['build_model']


def replace_conv2d_with_my_conv2d(net, ws_eps=None):
	if ws_eps is None:
		return

	for m in net.modules():
		to_update_dict = {}
		for name, sub_module in m.named_children():
			if isinstance(sub_module, nn.Conv2d) and sub_module.bias is None:
				# only replace conv2d layers that are followed by normalization layers (i.e., no bias)
				to_update_dict[name] = sub_module
		for name, sub_module in to_update_dict.items():
			m._modules[name] = WSConv2d(
				sub_module.in_channels, sub_module.out_channels, sub_module.kernel_size, sub_module.stride,
				sub_module.padding, sub_module.dilation, sub_module.groups, sub_module.bias is not None,
			)
			# load weight
			m._modules[name].load_state_dict(sub_module.state_dict())
			# load requires_grad
			m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
			if sub_module.bias is not None:
				m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
	# set ws_eps
	for m in net.modules():
		if isinstance(m, WSConv2d):
			m.ws_eps = ws_eps


def build_model(model_name, pretrained=True, num_classes=1000, input_size=224, tuning_method='full',  args=None, **kwargs):
    tuning_config = set_tuning_config(tuning_method, args)
    model = eval(model_name)(pretrained=pretrained, tuning_config=tuning_config, input_resolution=input_size, **kwargs)
    # reinitialize head
    model.head = LinearHead(model.num_features, num_classes, 0.2)# nn.Linear(model.num_features, num_classes)

    # freeze parameters if needed
    if tuning_method == 'full':
        # all parameters are trainable
        pass 
    elif tuning_method == 'prompt':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue

            if name.startswith('norm'):
                continue

            if 'tuning_module' in name:
                continue

            param.requires_grad = False
    elif tuning_method == 'adapter':
        raise NotImplementedError
    elif tuning_method == 'sidetune':
        raise NotImplementedError
    elif tuning_method == 'linear':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue
            
            if name.startswith('norm'):
                continue

            param.requires_grad = False
    elif tuning_method == 'norm':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue

            if 'bn' in name:
                continue

            if 'gn' in name:
                continue

            if 'norm' in name:
                continue
            
            # adjust last group norm
            if 'before_head' in name:
                continue

            param.requires_grad = False    
    elif tuning_method == 'bias':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue
            
            if name.startswith('norm'):
                continue

            if 'bias' in name:
                continue

            param.requires_grad = False
    elif tuning_method == 'conv_adapt' or tuning_method == 'repnet':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue
            
            if 'tuning_module' in name:
                continue
            
            # add a norm layer before average pooling
            if 'norm' in name:
                continue

            param.requires_grad = False
    elif tuning_method == 'conv_adapt_norm':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue
            
            if 'tuning_module' in name:
                continue

            if 'bn' in name:
                continue

            if 'gn' in name:
                continue

            if 'norm' in name:
                continue
            
            # adjust last group norm
            if 'before_head' in name:
                continue

            param.requires_grad = False    
    elif tuning_method == 'conv_adapt_bias' or tuning_method == 'repnet_bias':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue
            
            if 'tuning_module' in name:
                continue
                
            if 'bias' in name:
                continue
            
            # add a norm layer before average pooling
            if name.startswith('norm'):
                continue

            param.requires_grad = False
    
    if 'repnet' in tuning_method:
        replace_conv2d_with_my_conv2d(model, 1e-5)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")

    return model

