def compute_conv_flops(mod, input_shape, macs = False):
    
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
        flops *= 2
        
    return int(flops)
    

def compute_fc_flops(mod, macs = False):
    ft_in, ft_out =  mod.weight.data.shape
    flops = ft_in * ft_out
    
    if not macs:
        flops *= 2
        
    return int(flops)