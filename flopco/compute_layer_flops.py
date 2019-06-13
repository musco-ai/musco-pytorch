import torch

def compute_conv2d_flops(mod, input_shape = None, output_shape = None, macs = False):
    
    _, cin, h, w = input_shape
    w_cout, w_cin, w_h, w_w =  mod.weight.data.shape

    if mod.groups != 1:
        input_channels = 1
    else:
        assert cin == w_cin
        input_channels = w_cin

    output_channels = w_cout
    stride = mod.stride[0]

    flops = h * w * output_channels * input_channels * w_h * w_w / (stride**2)
    
    if not macs:
        flops_bias = output_shape[1:].numel() if mod.bias is not None else 0
        flops = 2 * flops + flops_bias
        
    return int(flops)
    

def compute_fc_flops(mod, input_shape = None, output_shape = None, macs = False):
    ft_in, ft_out =  mod.weight.data.shape
    flops = ft_in * ft_out
    
    if not macs:
        flops_bias = ft_out if mod.bias is not None else 0
        flops = 2 * flops + flops_bias
        
    return int(flops)

def compute_bn2d_flops(mod, input_shape = None, output_shape = None, macs = False):
    # subtract, divide, gamma, beta
    flops = 2 * input_shape[1:].numel()
    
    if not macs:
        flops *= 2
    
    return int(flops)


def compute_relu_flops(mod, input_shape = None, output_shape = None, macs = False):
    
    flops = 0
    if not macs:
        flops = input_shape[1:].numel()

    return int(flops)


def compute_maxpool2d_flops(mod, input_shape = None, output_shape = None, macs = False):

    flops = 0
    if not macs:
        flops = mod.kernel_size**2 * output_shape[1:].numel()

    return flops


def compute_avgpool2d_flops(mod, input_shape = None, output_shape = None, macs = False):

    flops = 0
    if not macs:
        flops = mod.kernel_size**2 * output_shape[1:].numel()

    return flops


def compute_softmax_flops(mod, input_shape = None, output_shape = None, macs = False):
    
    nfeatures = input_shape[1:].numel()
    
    total_exp = nfeatures # https://stackoverflow.com/questions/3979942/what-is-the-complexity-real-cost-of-exp-in-cmath-compared-to-a-flop
    total_add = nfeatures - 1
    total_div = nfeatures
    
    flops = total_div + total_exp
    
    if not macs:
        flops += total_add
        
    return flops
        
    