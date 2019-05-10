import numpy as np
import torch
from torch import nn


from .rank_selection import estimate_vbmf_ranks, estimate_rank_for_compression_rate


class SVDDecomposedLayer():
    def __init__(self, layer, layer_name,
                 rank = None,
                 pretrained = None,
                 vbmf_weaken_factor = None):

        self.layer_name = layer_name
        self.layer = layer
        self.pretrained = pretrained
       
        if  '__getitem__' in dir(self.layer):
            self.in_features = self.layer[0].in_features
            self.out_features = self.layer[1].out_features
        else:
            if self.layer._get_name() != 'Linear':
                raise AttributeError('only linear layer can be decomposed')
            self.in_features = self.layer.in_features
            self.out_features = self.layer.out_features
        
        self.weight, self.bias = self.get_weights_to_decompose()
        if rank == 0:
            self.is_vbmf = True
            self.vbmf_weaken_factor = vbmf_weaken_factor
            self.rank = estimate_vbmf_ranks(self.weight, self.vbmf_weaken_factor)
        else:
            self.is_vbmf = False
            rank = -rank
            if  '__getitem__' in dir(self.layer):
                self.rank = self.layer[0].out_features//rank
            else:
                self.rank = estimate_rank_for_compression_rate((self.out_features,
                                                                self.in_features),
                                                                rate = rank,
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
        if  '__getitem__' in dir(self.layer):
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
            
            if  '__getitem__' in dir(self.layer):
                w0_old = self.layer[0].weight.cpu().data
                w0 = np.dot(w0, w0_old)
                
            w0 = torch.FloatTensor(w0).contiguous()
            w1 = torch.FloatTensor(w1).contiguous()
            

        return [w0, w1], [None, bias]