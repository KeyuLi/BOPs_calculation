import torch.nn as nn
import torch
import numpy as np


def compute_bops(module, inp, out):
    if str(module) == str('HardBinaryConv()'):
        return compute_Conv2d_bops(module, inp, out)
    elif str(module) == str('BinaryActivation()'):
        return compute_ba_bops(module, inp, out)
    else:
        return 0
    pass


def compute_Conv2d_bops(module, inp, out):
    # Can have multiple inputs, getting the first one
    # assert isinstance(module, nn.Conv2d)
    # assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.shape[-2:]
    out_c, out_h, out_w = out.size()[1:]
    # groups = module.groups

    conv_per_position_bops = k_h * k_w * in_c * out_c
    active_elements_count = batch_size * out_h * out_w

    total_conv_bops = conv_per_position_bops * active_elements_count

    bias_bops = 0
    # if module.bias is not None:
    #     bias_bops = out_c * active_elements_count

    total_bops = total_conv_bops + bias_bops
    return total_bops

def compute_ba_bops(module, inp, out):
    batch_size = inp.size()[0]
    active_elements_count = batch_size

    for s in inp.size()[1:]:
        active_elements_count *= s

    return active_elements_count