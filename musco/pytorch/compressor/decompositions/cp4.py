import numpy as np
import torch
from torch import nn

import scipy

import tensorly as tl
from tensorly.decomposition import parafac_new
from tensorly.kruskal_tensor import kruskal_to_tensor

from musco.pytorch.compressor.rank_selection.estimator import estimate_rank_for_compression_rate

tl.set_backend('pytorch')


class CP4DecomposedLayer():
    def __init__(self, layer,\
                 layer_name,\
                 rank_selection,\
                 rank = None,\
                 pretrained = None,
                 param_reduction_rate = None):
        
        self.layer_name = layer_name
        self.layer = layer
            
        if  isinstance(self.layer, nn.Sequential):
            self.cin = self.layer[0].in_channels
            self.cout = self.layer[-1].out_channels
            
            self.kernel_size = (self.layer[1].kernel_size[0], self.layer[2].kernel_size[1]) 
            self.padding = (self.layer[1].padding[0], self.layer[2].padding[1])
            self.stride = (self.layer[1].stride[0], self.layer[2].stride[1])
           
        else:
            if not isinstance(self.layer, nn.Conv2d):
                raise AttributeError('only convolution layer can be decomposed')
            self.cin = self.layer.in_channels
            self.cout = self.layer.out_channels
            
            self.kernel_size = self.layer.kernel_size
            self.padding = self.layer.padding
            self.stride = self.layer.stride
            
        self.weight, self.bias = self.get_weights_to_decompose()
        
        
        if rank_selection == 'param_reduction':
            if  isinstance(self.layer, nn.Sequential):
                self.rank = estimate_rank_for_compression_rate((self.layer[1].out_channels,
                                                                 self.layer[1].in_channels,
                                                                 *self.kernel_size),
                                                                rate = param_reduction_rate,
                                                                key = 'cp4')
            else:
                self.rank = estimate_rank_for_compression_rate((self.cout, self.cin, *self.kernel_size),
                                            rate = param_reduction_rate,
                                            key = 'cp4')
        elif rank_selection == 'manual':
            self.rank = rank
        
        self.rank = int(self.rank)
            
        ##### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        weights, biases = self.get_cp_factors()        
        
        for j, (w, b)  in enumerate(zip(weights, biases)):
            self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).weight.data = w
            if b is not None:
                self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).bias.data = b
            else:
                self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).bias = None 
                
        self.layer = None
        self.weight = None
        self.bias = None
        
    def create_new_layers(self):
        layers = [] 
        layers.append(nn.Conv2d(in_channels=self.cin,
                                    out_channels=self.rank,
                                    kernel_size = (1, 1)))
        
        layers.append(nn.Conv2d(in_channels = self.rank, 
                                    out_channels=self.rank,
                                    kernel_size = (self.kernel_size[0], 1),
                                    groups = self.rank, 
                                    padding = (self.padding[0],0),
                                    stride = (self.stride[0], 1)))
                          
        layers.append(nn.Conv2d(in_channels = self.rank,
                                    out_channels=self.rank,
                                    kernel_size = (1, self.kernel_size[1]),
                                    groups = self.rank,
                                    padding = (0, self.padding[1]),
                                    stride = (1, self.stride[1])))
        
        layers.append(nn.Conv2d(in_channels = self.rank,
                                    out_channels = self.cout, 
                                    kernel_size = (1, 1)))
        return layers
    
    
    def get_weights_to_decompose(self):
        if isinstance(self.layer, nn.Sequential):
            w_cin = self.layer[0].weight.data
            w_h = self.layer[1].weight.data
            w_w = self.layer[2].weight.data
            w_cout = self.layer[3].weight.data

            try:
                bias = self.layer[2].bias.data
            except:
                bias = None

            f_h = torch.transpose(w_h.reshape(w_h.shape[0], w_h.shape[2]), 1, 0)
            f_w = torch.transpose(w_w.reshape(w_w.shape[0], w_w.shape[3]), 1, 0)
            f_cin = torch.transpose(w_cin.reshape(*w_cin.shape[:2]), 1, 0)
            f_cout = w_cout.reshape(*w_cout.shape[:2])

#             weight = ktensor([dtensor(f_cout), dtensor(f_cin), dtensor(f_h), dtensor(f_w)])
#             weight = torch.tensor(weight.totensor())
            weight = [f_cout, f_cin, f_h, f_w]

        else:
            weight = self.layer.weight.data
            try:
                bias = self.layer.bias.data
            except:
                bias = None
        return weight, bias
    
    
    def get_cp_factors(self):
        tl.set_backend('pytorch')
        bias = self.bias
        
        if isinstance(self.layer, nn.Sequential):
            # Tensorly case
            (f_cout, f_cin, f_h, f_w), _ = parafac_new(kruskal_to_tensor((None, self.weight)),\
                                                       self.rank,\
                                                       n_iter_max=50000,\
                                                       init='random',\
                                                       tol=1e-8,\
                                                       svd = None) 

            
        else:
            # Tensorly case
            (f_cout, f_cin, f_h, f_w), _ =  parafac_new(self.weight,\
                                               self.rank, n_iter_max=50000,\
                                               init='random',\
                                               tol=1e-8,\
                                               svd = None)
        
        
#         # Reshape factor matrices to 4D weight tensors
#         f_cin: (cin, rank) -> (rank, cin, 1, 1)
#         f_h: (h, rank) -> (rank, 1, h, 1)
#         f_w: (w, rank) -> (rank, 1, 1, w)
#         f_cout: (count, rank) -> (count, rank, 1, 1)

        # Pytorh case
        f_cin = f_cin.t().unsqueeze_(2).unsqueeze_(3).contiguous()
        f_h = f_h.t().unsqueeze_(1).unsqueeze_(3).contiguous()
        f_w = f_w.t().unsqueeze_(1).unsqueeze_(2).contiguous()
        f_cout = f_cout.unsqueeze_(2).unsqueeze_(3).contiguous()
        
        return [f_cin, f_h, f_w, f_cout], [None, None, None, bias]
