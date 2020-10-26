import numpy as np
import torch
from torch import nn
from argparse import Namespace

import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.kruskal_tensor import kruskal_to_tensor

from musco.pytorch.compressor.rank_estimation.estimator import estimate_rank_for_compression_rate
from .base import DecomposedLayer

tl.set_backend('pytorch')

class CP3DecomposedLayer(DecomposedLayer):
    """Convolutional layer with a kernel (kxk spacial size, k>1) represented in CP3 format.

    """
    
    def __init__(self,
                 layer,
                 layer_name,
                 **rank_kwargs):
        
        super(CP3DecomposedLayer, self).__init__(layer, layer_name)
        
        self.decomposition = 'cp3'
        self.min_rank = 2
        
        self.cin = None
        self.cout = None
        self.kernel_size = None
        self.padding = None
        self.stride = None
        self.device = None
        
        # Initialize layer parameters
        self.init_layer_params()
        self.init_device()
                
        # Estimate rank for tensor approximation and build new layers
        self.estimate_rank(**rank_kwargs)
        self.build_new_layers()
        
        # Compute weights for new layers, initialize new layers
        self.init_new_layers(*self.compute_new_weights(*self.extract_weights()))
                
        self.layer = None
        
        
    def init_layer_params(self):

        if  isinstance(self.layer, nn.Sequential):
            self.cin = self.layer[0].in_channels
            self.cout = self.layer[-1].out_channels

            self.kernel_size = self.layer[1].kernel_size 
            self.padding = self.layer[1].padding
            self.stride = self.layer[1].stride

        else:
            if not isinstance(self.layer, nn.Conv2d):
                raise AttributeError('Only convolutional layer can be decomposed using CP3')
            self.cin = self.layer.in_channels
            self.cout = self.layer.out_channels

            self.kernel_size = self.layer.kernel_size
            self.padding = self.layer.padding
            self.stride = self.layer.stride

            
    def estimate_rank(self, **rank_kwargs):
        rank_kwargs = Namespace(**rank_kwargs)

        if rank_kwargs.rank_selection == 'param_reduction':
            if  isinstance(self.layer, nn.Sequential):
                prev_rank = self.layer[0].out_channels
            else:
                prev_rank = None

            tensor_shape = (self.cout, self.cin, *self.kernel_size)
            
            self.rank =  estimate_rank_for_compression_rate(tensor_shape,
                                                               rate = rank_kwargs.param_reduction_rate,
                                                               tensor_format = self.decomposition,
                                                               prev_rank = prev_rank,
                                                               min_rank = self.min_rank)
        elif rank_kwargs.rank_selection == 'manual':
            self.rank =  rank_kwargs.manual_rank
            
        
    def extract_weights(self):
        if isinstance(self.layer, nn.Sequential):
            w_cin = self.layer[0].weight.data
            w_z = self.layer[1].weight.data
            w_cout = self.layer[-1].weight.data

            try:
                bias = self.layer[-1].bias.data
            except:
                bias = None

            f_z = w_z.reshape(w_z.shape[0], np.prod(w_z.shape[2:])).t()
            f_cin = w_cin.reshape(*w_cin.shape[:2]).t()
            f_cout = w_cout.reshape(*w_cout.shape[:2])

            weight = [f_cout, f_cin, f_z]

        else:
            weight = self.layer.weight.data
            weight = weight.reshape(*weight.shape[:2], -1)
            try:
                bias = self.layer.bias.data
            except:
                bias = None
                
        return weight, bias
    
    
    def compute_new_weights(self, weight, bias):
            
        if isinstance(self.layer, nn.Sequential):
            _, (f_cout, f_cin, f_z) = parafac(kruskal_to_tensor((None, weight)),
                                              self.rank,
                                              n_iter_max=5000,
                                              init='random',
                                              tol=1e-8,
                                              svd = None,
                                              cvg_criterion = 'rec_error') 

        else:                  
            _, (f_cout, f_cin, f_z) = parafac(weight,
                                              self.rank,
                                              n_iter_max=5000,
                                              init='random',
                                              tol=1e-8,
                                              svd = None,
                                              cvg_criterion = 'rec_error')
                
#         # Reshape factor matrices to 4D weight tensors
#         f_cin: (cin, rank) -> (rank, cin, 1, 1)
#         f_z: (z, rank) -> (rank, 1, h, w)
#         f_cout: (count, rank) -> (count, rank, 1, 1)
        
        # Pytorh case
        f_cin = f_cin.t().unsqueeze_(2).unsqueeze_(3).contiguous()
        f_z = torch.einsum('hwr->rhw', f_z.resize_((*self.kernel_size, self.rank))\
                          ).unsqueeze_(1).contiguous()
        f_cout = f_cout.unsqueeze_(2).unsqueeze_(3).contiguous()

        return [f_cin, f_z,  f_cout], [None, None,  bias]
    
        
    def build_new_layers(self):
            
        layers = [] 
        layers.append(nn.Conv2d(in_channels=self.cin,
                                    out_channels=self.rank,
                                    kernel_size = (1, 1)))
        
        layers.append(nn.Conv2d(in_channels = self.rank, 
                                    out_channels=self.rank,
                                    kernel_size = self.kernel_size,
                                    groups = self.rank, 
                                    padding = self.padding,
                                    stride = self.stride))
       
        layers.append(nn.Conv2d(in_channels = self.rank,
                                    out_channels = self.cout, 
                                    kernel_size = (1, 1)))
        
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(layers):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
    

