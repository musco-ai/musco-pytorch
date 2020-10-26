import numpy as np
import torch
from torch import nn
from argparse import Namespace

from musco.pytorch.compressor.rank_estimation.estimator import estimate_rank_for_compression_rate, estimate_vbmf_ranks
from .base import DecomposedLayer


class SVDDecomposedLayer(DecomposedLayer):
    """Fully connected layer with a weight represented in SVD format.
    
    """
    def __init__(self,
                 layer,
                 layer_name,
                 **rank_kwargs):
        super(SVDDecomposedLayer, self).__init__(layer, layer_name)
        
        self.decomposition = 'svd'
        self.min_rank = 8
        
        self.in_features = None
        self.out_features = None
        self.device = None
        
        # Initialize layer parameters
        self.init_layer_params()
        self.init_device()
                
        weight, bias = self.extract_weights()
        
        # Estimate rank for tensor approximation and build new layers
        self.estimate_rank(weight = weight, **rank_kwargs)
        self.build_new_layers()
        
        # Compute weights for new layers, initialize new layers
        self.init_new_layers(*self.compute_new_weights(weight, bias))
                
        self.layer = None
        weight = None
        bias = None

        
    def init_layer_params(self):
        if  isinstance(self.layer, nn.Sequential):
            self.in_features = self.layer[0].in_features
            self.out_features = self.layer[1].out_features
        else:
            if not isinstance(self.layer, nn.Linear):
                raise AttributeError('Only linear layer can be decomposed')
            self.in_features = self.layer.in_features
            self.out_features = self.layer.out_features
    
    
    def extract_weights(self):
        if  isinstance(self.layer, nn.Sequential):
#             weight = self.layer[1].weight.data @ self.layer[0].weight.data 
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
    
    
    def estimate_rank(self, weight = None, **rank_kwargs):
        rank_kwargs = Namespace(**rank_kwargs)
        
        if rank_kwargs.rank_selection == 'vbmf':
            self.rank = estimate_vbmf_ranks(weight,
                                            rank_kwargs.vbmf_weaken_factor,
                                            min_rank = self.min_rank)
        
        elif rank_kwargs.rank_selection == 'manual':
            self.rank = rank_kwargs.manual_rank
            
        elif rank_kwargs.rank_selection == 'param_reduction':
            if  isinstance(self.layer, nn.Sequential):
                prev_rank = self.layer[0].out_features
            else:
                prev_rank = None

            self.rank = estimate_rank_for_compression_rate((self.out_features, self.in_features),
                                                           rate = rank_kwargs.param_reduction_rate,
                                                           tensor_format = self.decomposition,
                                                           prev_rank = prev_rank,
                                                           min_rank = self.min_rank)
    
    
    def compute_new_weights(self, weight, bias):
        weights = weight.cpu()
        if bias is not None:
            bias = bias.cpu()

        U, S, Vt = np.linalg.svd(weights.data.numpy(), full_matrices=False)

        w0 = np.dot(np.diag(np.sqrt(S[0:self.rank])),Vt[0:self.rank, :])
        w1 = np.dot(U[:, 0:self.rank], np.diag(np.sqrt(S[0:self.rank])))

        if  isinstance(self.layer, nn.Sequential):
            w0_old = self.layer[0].weight.cpu().data
            w0 = np.dot(w0, w0_old)

        w0 = torch.FloatTensor(w0).contiguous()
        w1 = torch.FloatTensor(w1).contiguous()
            

        return [w0, w1], [None, bias]
    
    
    def build_new_layers(self):                       

        layers = [] 
        layers.append(nn.Linear(in_features = self.in_features, 
                                out_features = self.rank,
                                 bias = False))
        layers.append(nn.Linear(in_features = self.rank, 
                                out_features = self.out_features))
        
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(layers):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)


class SVDDecomposedConvLayer(DecomposedLayer):
    """Convolutional layer with a kernel (1x1 spacial size) represented in SVD format.
    
    """
    def __init__(self,
                 layer,
                 layer_name,
                 **rank_kwargs):
        
        super(SVDDecomposedConvLayer, self).__init__(layer, layer_name)
        
        self.decomposition = 'svd'
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
        self.estimate_rank(weight = weight, **rank_kwargs)
        self.build_new_layers()
        
        # Compute weights for new layers, initialize new layers
        self.init_new_layers(*self.compute_new_weights(weight, bias))
                
        self.layer = None
        weight = None
        bias = None

        
    def init_layer_params(self):
        if  isinstance(self.layer, nn.Sequential):
            self.in_channels = self.layer[0].in_channels
            self.out_channels = self.layer[1].out_channels

            self.padding = self.layer[1].padding
            self.stride = self.layer[1].stride
        else:
            if not isinstance(self.layer, nn.Conv2d):
                raise AttributeError('Only convolutional layer can be decomposed')
            self.in_channels = self.layer.in_channels
            self.out_channels = self.layer.out_channels

            self.padding = self.layer.padding
            self.stride = self.layer.stride
    
    
    def extract_weights(self):
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
 

    def estimate_rank(self, weight = None, **rank_kwargs):
        rank_kwargs = Namespace(**rank_kwargs)
        
        if rank_kwargs.rank_selection == 'vbmf':
            self.rank = estimate_vbmf_ranks(weight,
                                            rank_kwargs.vbmf_weaken_factor,
                                            min_rank = self.min_rank)
            
        elif rank_kwargs.rank_selection == 'manual':
            self.rank = rank_kwargs.manual_rank
            
        elif rank_kwargs.rank_selection == 'param_reduction':
            if  isinstance(self.layer, nn.Sequential):
                prev_rank = self.layer[0].out_channels
            else:
                prev_rank = None

            self.rank = estimate_rank_for_compression_rate((self.out_channels, self.in_channels),
                                                           rate = rank_kwargs.param_reduction_rate,
                                                           tensor_format = self.decomposition,
                                                           prev_rank = prev_rank,
                                                           min_rank = self.min_rank)
            
        
    def compute_new_weights(self, weight, bias):
        weights = weight.cpu()
        if bias is not None:
            bias = bias.cpu()

        U, S, Vt = np.linalg.svd(weights.data.numpy(), full_matrices=False)

        w0 = np.dot(np.diag(np.sqrt(S[0:self.rank])),Vt[0:self.rank, :])
        w1 = np.dot(U[:, 0:self.rank], np.diag(np.sqrt(S[0:self.rank])))

        if  isinstance(self.layer, nn.Sequential):
            w0_old = self.layer[0].weight[:,:,0,0].cpu().data
            w0 = np.dot(w0, w0_old)

        w0 = torch.FloatTensor(w0[:,:, np.newaxis, np.newaxis]).contiguous()
        w1 = torch.FloatTensor(w1[:,:, np.newaxis, np.newaxis]).contiguous()

        return [w0, w1], [None, bias]

    
    def build_new_layers(self):                       

        layers = [] 
        layers.append(nn.Conv2d(in_channels = self.in_channels, 
                                out_channels = self.rank,
                                kernel_size = 1, 
                                bias = False))
        layers.append(nn.Conv2d(in_channels = self.rank,  
                                out_channels = self.out_channels,
                                kernel_size = 1,
                                padding = self.padding,
                                stride = self.stride))
        
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(layers):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)           