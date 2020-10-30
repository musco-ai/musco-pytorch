import numpy as np
import torch
from torch import nn
from argparse import Namespace

from sktensor import dtensor, tucker
from musco.pytorch.compressor.rank_estimation.estimator import estimate_rank_for_compression_rate, estimate_vbmf_ranks
from .base import DecomposedLayer

class Tucker2DecomposedLayer(nn.Module, DecomposedLayer):
    """Convolutional layer with a kernel (kxk spacial size, k>1) represented in Tucker2 format.
    
    References
    ----------
    .. [1] Kim, Y.D. et al. (2016). "Compression of deep convolutional neural networks for fast and low power mobile applications". Proceedings of the International Conference on Learning Representations.
    
    """
    
    def __init__(self, layer, layer_name, algo_kwargs={}, **compr_kwargs):
        nn.Module.__init__(self)
        DecomposedLayer.__init__(self, layer, layer_name, algo_kwargs=algo_kwargs)
        
        assert compr_kwargs['decomposition'] == 'tucker2'
        self.min_rank = 8
        
        self.cin = None
        self.cout = None
        self.kernel_size = None
        self.padding = None
        self.stride = None
        self.device = None
                
        # Initialize layer parameters
        self.init_layer_params()
        self.init_device()
                
        weight, bias = self.extract_weights()
        
        # Estimate rank for tensor approximation and build new layers
        self.estimate_rank(weight = weight, **compr_kwargs)
        self.build_new_layers()
        
        # Compute weights for new layers, initialize new layers
        self.init_new_layers(*self.compute_new_weights(weight, bias, algo_kwargs))
                
        self.layer = None
        self.__delattr__('layer')
        weight = None
        bias = None
    
    
    def init_layer_params(self):
        if  isinstance(self.layer, nn.Sequential):
            self.cin = self.layer[0].in_channels
            self.cout = self.layer[2].out_channels

            self.kernel_size = self.layer[1].kernel_size
            self.padding = self.layer[1].padding
            self.stride = self.layer[1].stride

        else:
            if not isinstance(self.layer, nn.Conv2d):
                raise AttributeError('Only convolutional layer can be decomposed')
            self.cin = self.layer.in_channels
            self.cout = self.layer.out_channels

            self.kernel_size = self.layer.kernel_size
            self.padding = self.layer.padding
            self.stride = self.layer.stride
    
    
    def extract_weights(self):
        if  isinstance(self.layer, nn.Sequential):
            weight = self.layer[1].weight.data
            weight = weight.reshape(*weight.shape[:2], -1)
            try:
                bias = self.layer[2].bias.data
            except:
                bias = None
        else:
            weight = self.layer.weight.data
            weight = weight.reshape(*weight.shape[:2], -1)
            try:
                bias = self.layer.bias.data
            except:
                bias = None
        return weight, bias
    
    
    def estimate_rank(self, weight = None, **compr_kwargs):
        compr_kwargs = Namespace(**compr_kwargs)
        
        if compr_kwargs.rank_selection == 'vbmf':
            self.rank = estimate_vbmf_ranks(weight,
                                            compr_kwargs.vbmf_weakenen_factor,
                                            min_rank = self.min_rank)
            
        elif compr_kwargs.rank_selection == 'manual':
            i = compr_kwargs.curr_compr_iter
            self.rank = compr_kwargs.manual_rank[i]  #rank = [rank_cout, rank_cin]
            
        elif compr_kwargs.rank_selection == 'param_reduction':
            if  isinstance(self.layer, nn.Sequential):
                prev_rank = (self.layer[1].out_channels, self.layer[1].in_channels)
            else:
                prev_rank = None
            tensor_shape = (self.cout, self.cin, *self.kernel_size)
                
            self.rank = estimate_rank_for_compression_rate(tensor_shape,
                                                            rate = compr_kwargs.param_reduction_rate,
                                                            tensor_format = compr_kwargs.decomposition,
                                                            prev_rank = prev_rank,
                                                            min_rank = self.min_rank)
    
        
    def compute_new_weights(self, weight, bias, algo_kwargs={}):
        weights = dtensor(weight.cpu())
        if bias is not None:
            bias = bias.cpu()

        core, (U_cout, U_cin, U_dd) = tucker.hooi(weights,
                                                  [self.rank[0],
                                                   self.rank[1],
                                                   weights.shape[-1]], **algo_kwargs)
        core = core.dot(U_dd.T)

        w_cin = np.array(U_cin)
        w_core = np.array(core)
        w_cout = np.array(U_cout)

        if isinstance(self.layer, nn.Sequential):
            w_cin_old = self.layer[0].weight.cpu().data
            w_cout_old = self.layer[2].weight.cpu().data

            U_cin_old = np.array(torch.transpose(w_cin_old.reshape(w_cin_old.shape[:2]), 1, 0))
            U_cout_old = np.array(w_cout_old.reshape(w_cout_old.shape[:2]))

            w_cin = U_cin_old.dot(U_cin)
            w_cout = U_cout_old.dot(U_cout)

        w_cin = torch.FloatTensor(np.reshape(w_cin.T, [self.rank[1], self.cin, 1, 1])).contiguous()
        w_core = torch.FloatTensor(np.reshape(w_core, [self.rank[0], self.rank[1], *self.kernel_size])).contiguous()
        w_cout = torch.FloatTensor(np.reshape(w_cout, [self.cout, self.rank[0], 1, 1])).contiguous()

        return [w_cin, w_core,  w_cout], [None, None,  bias]

    
    def build_new_layers(self):                       

        layers = [] 
        layers.append(nn.Conv2d(in_channels=self.cin,
                                    out_channels=self.rank[1],
                                    kernel_size = (1, 1)))

        layers.append(nn.Conv2d(in_channels = self.rank[1], 
                                    out_channels=self.rank[0],
                                    kernel_size = self.kernel_size,
                                    groups = 1, 
                                    padding = self.padding,
                                    stride = self.stride))

        layers.append(nn.Conv2d(in_channels = self.rank[0],
                                    out_channels = self.cout, 
                                    kernel_size = (1, 1)))
        
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(layers):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
            
    
    def forward(self, x):
        
        x = self.new_layers(x)
        return x
