import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def make_divisible(v, divisor, min_val=None):
	"""
	This function is taken from the original tf repo.
	It ensures that all layers have a channel number that is divisible by 8
	It can be seen here:
	https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
	:param v:
	:param divisor:
	:param min_val:
	:return:
	"""
	if min_val is None:
		min_val = divisor
	new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def build_activation(act_func, inplace=True):
	if act_func == 'relu':
		return nn.ReLU(inplace=inplace)
	elif act_func == 'relu6':
		return nn.ReLU6(inplace=inplace)
	elif act_func == 'tanh':
		return nn.Tanh()
	elif act_func == 'sigmoid':
		return nn.Sigmoid()
	elif act_func is None or act_func == 'none':
		return None
	else:
		raise ValueError('do not support: %s' % act_func)


def init_models(net, model_init='he_fout'):
	"""
		Conv2d,
		BatchNorm2d, BatchNorm1d, GroupNorm
		Linear,
	"""
	if isinstance(net, list):
		for sub_net in net:
			init_models(sub_net, model_init)
		return
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			if model_init == 'he_fout':
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif model_init == 'he_fin':
				n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			else:
				raise NotImplementedError
			if m.bias is not None:
				m.bias.data.zero_()
		elif type(m) in [nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm]:
			m.weight.data.fill_(1)
			m.bias.data.zero_()
		elif isinstance(m, nn.Linear):
			stdv = 1. / math.sqrt(m.weight.size(1))
			m.weight.data.uniform_(-stdv, stdv)
			if m.bias is not None:
				m.bias.data.zero_()

def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2


class ProgramModule(nn.Module):
    def __init__(self, input_size=224, in_channels=3, out_channels=3,
                    expand=1.0, kernel_size=5, act_func='relu', n_groups=2,
                    downsample_ratio=2, upsample_ratio=2, upsample_type='bilinear', stride=1):
        super(ProgramModule, self).__init__()
        self.input_size = input_size
        if downsample_ratio is not None:
            upsample_ratio = downsample_ratio//upsample_ratio
        if self.input_size == 7:
            kernel_size = 3
        self.encoder_config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'expand': expand,
            'kernel_size': kernel_size,
            'act_func': act_func,
            'n_groups': n_groups,
            'downsample_ratio': downsample_ratio,
            'upsample_type': upsample_type,
            'upsample_ratio': upsample_ratio,
            'stride': stride,
        }

        padding = get_same_padding(kernel_size)
        if downsample_ratio is None:
            pooling = nn.AvgPool2d(2, 2, 0)
        else:
            pooling = nn.AvgPool2d(downsample_ratio, downsample_ratio, 0)
        #only for resnet
        # expand = 1/4
        # if out_channels == 64:
        #     num_mid = 64
        # else:
        #     num_mid = make_divisible(int(in_channels * expand), divisor=MyNetwork.CHANNEL_DIVISIBLE)

        num_mid = make_divisible(int(in_channels * expand), divisor=8)

        self.encoder = nn.Sequential(OrderedDict({
            'pooling': pooling,
            'conv1': nn.Conv2d(in_channels, num_mid, kernel_size, stride, padding, groups=n_groups, bias=False),
            'bn1': nn.BatchNorm2d(num_mid, eps=0.001),
            'act': build_activation(act_func),
            'conv2': nn.Conv2d(num_mid, out_channels, 1, 1, 0, bias=False),
            'final_bn': nn.BatchNorm2d(out_channels, eps=0.001),
        }))

        # initialize
        init_models(self.encoder)
        self.encoder.final_bn.weight.data.zero_()

    def forward(self, x):
        
        encoder_x = self.encoder(x)
        if self.encoder_config['upsample_ratio'] is not None:
            encoder_x = F.upsample(encoder_x, (x.shape[2]//self.encoder_config['upsample_ratio'], x.shape[3]//self.encoder_config['upsample_ratio']),
                                            mode=self.encoder_config['upsample_type'])
        return encoder_x