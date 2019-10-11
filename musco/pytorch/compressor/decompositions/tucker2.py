import numpy as np
import torch
from torch import nn

from sktensor import dtensor, tucker
from musco.pytorch.compressor.rank_selection.estimator import estimate_rank_for_compression_rate, estimate_vbmf_ranks

class Tucker2DecomposedLayer():
    def __init__(self, layer, layer_name,
                 rank_selection,
                 ranks = None,
                 pretrained = None,
                 vbmf_weaken_factor = None,
                 param_reduction_rate = None):
        """
        rank_selection: str, 'vbmf'/'param_reduction'/'manual'
        """

        self.layer_name = layer_name
        self.layer = layer
        self.pretrained = pretrained
        
        if  isinstance(self.layer, nn.Sequential):
            self.cin = self.layer[0].in_channels
            self.cout = self.layer[2].out_channels
            
            self.kernel_size = self.layer[1].kernel_size
            self.padding = self.layer[1].padding
            self.stride = self.layer[1].stride
#             print('Sequential, cin:{}, cout: {}'.format(self.cin, self.cout))
           
        else:
            if not isinstance(self.layer, nn.Conv2d):
                raise AttributeError('only convolution layer can be decomposed')
            self.cin = self.layer.in_channels
            self.cout = self.layer.out_channels
            
            self.kernel_size = self.layer.kernel_size
            self.padding = self.layer.padding
            self.stride = self.layer.stride
#             print('Conv, cin:{}, cout: {}'.format(self.cin, self.cout))
           
        self.weight, self.bias = self.get_weights_to_decompose()
        
        if rank_selection == 'vbmf':
            self.ranks = estimate_vbmf_ranks(self.weight, vbmf_weaken_factor)
            
        elif rank_selection == 'manual':
            # self.ranks = [rank_cout, rank_cin]
            self.ranks = ranks
            
        elif rank_selection == 'param_reduction':
                if  isinstance(self.layer, nn.Sequential):
                    self.ranks = estimate_rank_for_compression_rate((self.layer[1].out_channels,\
                                                                     self.layer[1].in_channels,\
                                                                     *self.kernel_size),\
                                                                    rate = param_reduction_rate,\
                                                                    key = 'tucker2')
                else:
                    self.ranks = estimate_rank_for_compression_rate((self.cout,\
                                                                     self.cin,\
                                                                     *self.kernel_size),\
                                                                    rate = param_reduction_rate,\
                                                                    key = 'tucker2')
        #print(self.ranks)

        ##### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        weights, biases = self.get_tucker_factors()        
        
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
                                    out_channels=self.ranks[1],
                                    kernel_size = (1, 1)))

        layers.append(nn.Conv2d(in_channels = self.ranks[1], 
                                    out_channels=self.ranks[0],
                                    kernel_size = self.kernel_size,
                                    groups = 1, 
                                    padding = self.padding,
                                    stride = self.stride))

        layers.append(nn.Conv2d(in_channels = self.ranks[0],
                                    out_channels = self.cout, 
                                    kernel_size = (1, 1)))
        return layers
    
    
    def get_weights_to_decompose(self):
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
    
        
    def get_tucker_factors(self):
        if self.pretrained is not None:
            raise AttributeError('Not implemented')
        else:
            weights = dtensor(self.weight.cpu())
            if self.bias is not None:
                bias = self.bias.cpu()
            else:
                bias = self.bias
            
            core, (U_cout, U_cin, U_dd) = tucker.hooi(weights,
                                                      [self.ranks[0],
                                                       self.ranks[1],
                                                       weights.shape[-1]], init='nvecs')
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

        w_cin = torch.FloatTensor(np.reshape(w_cin.T, [self.ranks[1], self.cin, 1, 1])).contiguous()
        w_core = torch.FloatTensor(np.reshape(w_core, [self.ranks[0], self.ranks[1], *self.kernel_size])).contiguous()
        w_cout = torch.FloatTensor(np.reshape(w_cout, [self.cout, self.ranks[0], 1, 1])).contiguous()

        return [w_cin, w_core,  w_cout], [None, None,  bias]
