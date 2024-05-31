from .prompter import PadPrompter
from .conv_adapter import ConvAdapter, LinearAdapter
from .program_module import ProgramModule


def set_tuning_config(tuning_method, args):
    tuning_config = None
    if tuning_method == 'conv_adapt' or tuning_method == 'conv_adapt_norm' or tuning_method == 'conv_adapt_bias':
        tuning_config = {
            'method': tuning_method,
            'kernel_size': args.kernel_size,
            'adapt_size': args.adapt_size,
            'adapt_scale': args.adapt_scale,
        }
    elif tuning_method == 'prompt':
        tuning_config = {
            'method': tuning_method,
            'prompt_size': args.prompt_size,
        }
    elif tuning_method == 'full' or tuning_method == 'linear' or tuning_method == 'norm' or tuning_method == 'repnet' or tuning_method == 'repnet_bias' or tuning_method == 'bias':
        tuning_config = {
            'method': tuning_method,
        }
    else:
        raise NotImplementedError
    return tuning_config
