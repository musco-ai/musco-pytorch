import numpy as np
import torch
from torch import nn

from musco.pytorch.compressor.rank_selection.estimator import estimate_rank_for_compression_rate, estimate_vbmf_ranks


class SVDDecomposedLayer():
    def __init__(self, layer, layer_name,
                 rank_selection,
                 rank = None,
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
            self.in_features = self.layer[0].in_features
            self.out_features = self.layer[1].out_features
        else:
            if not isinstance(self.layer, nn.Linear):
                raise AttributeError('only linear layer can be decomposed')
            self.in_features = self.layer.in_features
            self.out_features = self.layer.out_features
        
        self.weight, self.bias = self.get_weights_to_decompose()
        
        if rank_selection == 'vbmf':
            self.rank = estimate_vbmf_ranks(self.weight, vbmf_weaken_factor)
        
        elif rank_selection == 'manual':
            self.rank = rank
            
        elif rank_selection == 'param_reduction':
            if  isinstance(self.layer, nn.Sequential):
                self.rank = self.layer[0].out_features//param_reduction_rate
            else:
                self.rank = estimate_rank_for_compression_rate((self.out_features,
                                                                self.in_features),
                                                                rate = param_reduction_rate,
                                                                key = 'svd')
        ##### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        weights, biases = self.get_svd_factors()        
        
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
        layers.append(nn.Linear(in_features = int(self.in_features), 
                                out_features = self.rank,
                                 bias = False))
        layers.append(nn.Linear(in_features = int(self.rank), 
                                out_features = self.out_features))
        return layers
    
    def get_weights_to_decompose(self):
        if  isinstance(self.layer, nn.Sequential):
            #weight = self.layer[1].weight.data @ self.layer[0].weight.data 
            weight = self.layer[1].weight.data
            try:
                bias = self.layer[1].bias.data
            except:
                bias = None
        else:
            weight = self.layer.weight.data
            try:
                bias = self.layer.bias.data
            except:
                bias = None
        return weight, bias
    
        
    def get_svd_factors(self):
        if self.pretrained is not None:
            raise AttributeError('Not implemented')
        else:
            weights = self.weight.cpu()
            if self.bias is not None:
                bias = self.bias.cpu()
            else:
                bias = self.bias
                
            U, S, Vt = np.linalg.svd(weights.data.numpy(), full_matrices=False)
            
            w0 = np.dot(np.diag(np.sqrt(S[0:self.rank])),Vt[0:self.rank, :])
            w1 = np.dot(U[:, 0:self.rank], np.diag(np.sqrt(S[0:self.rank])))
            
            if  isinstance(self.layer, nn.Sequential):
                w0_old = self.layer[0].weight.cpu().data
                w0 = np.dot(w0, w0_old)
                
            w0 = torch.FloatTensor(w0).contiguous()
            w1 = torch.FloatTensor(w1).contiguous()
            

        return [w0, w1], [None, bias]

    


class SVDDecomposedConvLayer():
    def __init__(self, layer, layer_name,
                 rank_selection,
                 rank = None,
                 pretrained = None,
                 vbmf_weaken_factor = None,
                 param_reduction_rate = None):

        self.layer_name = layer_name
        self.layer = layer
        self.pretrained = pretrained
        
        #print(layer)
       
        if  isinstance(self.layer, nn.Sequential):
            self.in_channels = self.layer[0].in_channels
            self.out_channels = self.layer[1].out_channels
            
            self.padding = self.layer[1].padding
            self.stride = self.layer[1].stride
        else:
            if not isinstance(self.layer, nn.Conv2d):
                raise AttributeError('only conv layer can be decomposed')
            self.in_channels = self.layer.in_channels
            self.out_channels = self.layer.out_channels
            
            self.padding = self.layer.padding
            self.stride = self.layer.stride
        
        self.weight, self.bias = self.get_weights_to_decompose()
        
        if rank_selection == 'vbmf':
            self.rank = estimate_vbmf_ranks(self.weight, vbmf_weaken_factor)
        elif rank_selection == 'manual':
            self.rank = int(rank)
        elif rank_selection == 'param_reduction':
            if  isinstance(self.layer, nn.Sequential):
                self.rank = self.layer[0].out_channels//param_reduction_rate
            else:
                self.rank = estimate_rank_for_compression_rate((self.out_channels,
                                                                self.in_channels),
                                                                rate = param_reduction_rate,
                                                                key = 'svd')
        ##### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        weights, biases = self.get_svd_factors()        
        
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
        layers.append(nn.Conv2d(in_channels = int(self.in_channels), 
                                out_channels = int(self.rank),
                                kernel_size = 1, 
                                bias = False))
        layers.append(nn.Conv2d(in_channels = int(self.rank),  
                                out_channels = self.out_channels,
                                kernel_size = 1,
                                padding = self.padding,
                                stride = self.stride))
        return layers
    
    def get_weights_to_decompose(self):
        if  isinstance(self.layer, nn.Sequential):
            #weight = self.layer[1].weight.data @ self.layer[0].weight.data 
            weight = self.layer[1].weight.data
            try:
                bias = self.layer[1].bias.data
            except:
                bias = None
        else:
            weight = self.layer.weight.data
            try:
                bias = self.layer.bias.data
            except:
                bias = None
        return weight[:,:,0,0], bias
    
        
    def get_svd_factors(self):
        if self.pretrained is not None:
            raise AttributeError('Not implemented')
        else:
            weights = self.weight.cpu()
            if self.bias is not None:
                bias = self.bias.cpu()
            else:
                bias = self.bias
                
            U, S, Vt = np.linalg.svd(weights.data.numpy(), full_matrices=False)
            
            w0 = np.dot(np.diag(np.sqrt(S[0:self.rank])),Vt[0:self.rank, :])
            w1 = np.dot(U[:, 0:self.rank], np.diag(np.sqrt(S[0:self.rank])))
            
            if  isinstance(self.layer, nn.Sequential):
                w0_old = self.layer[0].weight[:,:,0,0].cpu().data
                w0 = np.dot(w0, w0_old)
                
            w0 = torch.FloatTensor(w0[:,:, np.newaxis, np.newaxis]).contiguous()
            w1 = torch.FloatTensor(w1[:,:, np.newaxis, np.newaxis]).contiguous()
            

        return [w0, w1], [None, bias]
