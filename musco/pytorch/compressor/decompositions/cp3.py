import numpy as np
import torch
from torch import nn

import scipy

import tensorly as tl
from tensorly.decomposition import parafac_new
from tensorly.kruskal_tensor import kruskal_to_tensor

from musco.pytorch.compressor.rank_selection.estimator import estimate_rank_for_compression_rate

tl.set_backend('pytorch')

class CP3DecomposedLayer():
    def __init__(self, layer, layer_name,\
                 rank_selection,\
                 rank=None,\
                 pretrained = None,\
                 param_reduction_rate = None):
        
        
        self.layer_name = layer_name
        self.layer = layer
        self.pretrained = pretrained
        
        
        if  isinstance(self.layer, nn.Sequential):
            self.cin = self.layer[0].in_channels
            self.cout = self.layer[-1].out_channels
            
            self.kernel_size = self.layer[1].kernel_size 
            self.padding = self.layer[1].padding
            self.stride = self.layer[1].stride
            
            self.device = self.layer[0].weight.device
           
        else:
            if not isinstance(self.layer, nn.Conv2d):
                raise AttributeError('only convolution layer can be decomposed')
            self.cin = self.layer.in_channels
            self.cout = self.layer.out_channels
            
            self.kernel_size = self.layer.kernel_size
            self.padding = self.layer.padding
            self.stride = self.layer.stride
            
            self.device = self.layer.weight.device
                
        self.weight, self.bias = self.get_weights_to_decompose()
        
#         print("KERNEL SIZE", self.kernel_size, type(self.kernel_size[0]), type(self.kernel_size[1]))
        
        if rank_selection == 'param_reduction':
            if  isinstance(self.layer, nn.Sequential):
                self.rank = estimate_rank_for_compression_rate((self.layer[1].out_channels,
                                                                 self.layer[1].in_channels,
                                                                 *self.kernel_size),
                                                                rate = param_reduction_rate,
                                                                key = 'cp3')
            else:
                self.rank = estimate_rank_for_compression_rate((self.cout, self.cin, *self.kernel_size),
                                            rate = param_reduction_rate,
                                            key = 'cp3')
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
                                    kernel_size = self.kernel_size,
                                    groups = self.rank, 
                                    padding = self.padding,
                                    stride = self.stride))
       
        layers.append(nn.Conv2d(in_channels = self.rank,
                                    out_channels = self.cout, 
                                    kernel_size = (1, 1)))
        
        return layers
    
    
    def get_weights_to_decompose(self):
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
    
    def get_cp_factors(self):

        if self.pretrained is not None:
            mat_dict = scipy.io.loadmat(self.pretrained)
            
            if mat_dict['R'][0][0] != self.rank:
                print('WRONG FACTORS, do not correspond to desired rank')
                
            PU_z, PU_cout, PU_cin = [Ui[0] for Ui in mat_dict['P_bals_epc_U']]
            Plmbda = mat_dict['P_bals_epc_lambda'].ravel()
            
            f_cin = np.array(PU_cin) 
            f_cout = np.array(PU_cout)
            f_z = (np.array(PU_z)*(Plmbda))
            
        else:
            bias = self.bias
            if isinstance(self.layer, nn.Sequential):
                # Tensorly case
                (f_cout, f_cin, f_z), _ = parafac_new(kruskal_to_tensor((None, self.weight)),\
                                                       self.rank,\
                                                       n_iter_max=50000,\
                                                       init='random',\
                                                       tol=1e-8,\
                                                       svd = None) 
                
            else:                  
                  # Tensorly case
                (f_cout, f_cin, f_z), _ =  parafac_new(self.weight,\
                                                       self.rank, n_iter_max=50000,\
                                                       init='random',\
                                                       tol=1e-8,\
                                                       svd = None)
                
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
    
