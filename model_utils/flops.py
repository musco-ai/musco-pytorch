import torch.nn as nn
import numpy as np

def compute_layer_flops(layer, input_shape, channels=1., outputs=1.):
    s = input_shape
    p = layer.weight.data.shape
    
    if isinstance(layer, nn.Conv2d):
        if layer.groups != 1:
            channels = 1.
        else:
            assert s[1] == p[1]
            channels *= p[1]
        outputs *= p[0]
            
        stride = layer.stride[0]
        c = s[2]*s[3]*outputs*channels*p[2]*p[3]/(stride **2)
    elif isinstance(layer, nn.Linear):
        c = p[0]*p[1]
    else:
        pass
    return int(c)


def compute_model_flops(model, input_shapes_dict, verbose = False):
    conv_flops_dict = {}
    fc_flops_dict = {}

    for mname, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_flops_dict[mname] = compute_layer_flops(m, input_shapes_dict[mname])
        elif isinstance(m, nn.Linear):
            fc_flops_dict[mname] = compute_layer_flops(m, input_shapes_dict[mname])
            
    total_flops = {'conv': sum(conv_flops_dict.values()),
                   'fc': sum(fc_flops_dict.values())}
    if verbose:
        for (k,v) in dict(**conv_flops_dict, **fc_flops_dict).items():
            print(k, np.round(v/sum(total_flops.values()), decimals = 2))
        print()
        
    return total_flops, conv_flops_dict, fc_flops_dict
