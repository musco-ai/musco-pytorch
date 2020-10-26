import torch.nn as nn
import abc

class DecomposedLayer(object, metaclass = abc.ABCMeta):
    """Base class to build a linear layer, whose weight is represented in a factorized format.
    
    Attributes
    ----------
    layer_name : str
        A name of both initial layer and decomposed layer.
    rank : int or iterable
        A rank of the tensor decomposition, which is used to approximate the layer's weight.
    new_layers : nn.Sequential
        A sequence of layers (decomposed layer) that replaces the initial layer after the compression step is done.
        
    See Also
    --------
    .cp3.CP3DecomposedLayer
    .cp4.CP4DecomposedLayer
    .tucker2.Tucker2DecomposedLayer
    .svd_layer.SVDDecomposedLayer
    .svd_layer.SVDDecomposedConvLayer
    """
    
    def __init__(self,
                 layer,
                 layer_name,
                 rank_selection = None,
                 manual_rank = None,
                 vbmf_weaken_factor = None,
                 param_reduction_rate = None):
        """
        
        Parameters
        ----------
        layer : nn.Conv2D or nn.Linear or nn.Sequential
            A layer to compress using a low-rank tensor approximation.
                - If `layer` is nn.Conv2D or nn.Linear, then the layer is compressed for the first time.
                - If `layer` is nn.Sequential, then the layer is already represented in a factorized format and compressed further. 
            Depending on the `layer` type, different approximations are applied.    
                - If `layer` is nn.Conv2D with 1x1 spacial kernel or nn.Linear, then it is compressed by approximating the weight with truncated SVD.
                - If `layer` is nn.Conv2D with kxk (k>1) spacial kernel, then it is compressed by approximating the weight with Tucker2, CP3 or CP4 decomposition.
                - If `layer` is nn.Sequential, then it is further compressed using the same decomposition as during the first compression.
        layer_name : str
            A name of the `layer`.
        rank_selection : {'manual', 'param_reduction', 'vbmf'}
           A method to estimate the rank of the tensor decomposition.
        manual_rank : int or iterable or None
                - if `rank_selection` != 'manual', then `manual_rank` is -1.
                - Else, `manual_rank` equals *(int)* for CP3/CP4/SVD and *(iterable)* for Tucker2.
        param_reduction_rate : float or None
                - If `rank_selection` != 'param_reduction', then `param_reduction_rate` is None.
                - Else, `param_reduction` is a reduction factor by which the number of layer's parameters decrease after the compression step.
        vbmf_weaken_factor : float or None
                - If `rank_selection` != 'vbmf', then `vbmf_weaken_factor` is None.
                - Else, `vbmf_weaken_factor` is a weakenen factor used to increase tensor rank found via EVMBF.
        """
        
        self.layer_name = layer_name
        self.layer = layer

        self.rank = None
        self.new_layers = None
    
    @abc.abstractmethod
    def init_layer_params(self):
        """Sets parameters of an uncompressed layer.
        
        The following attributes are set,
            - when nn.Conv2d is compressed:
                - `cin` - input channels,
                - `cout` - output channels,
                - `kernel_size` - kernel spacial size,
                - `padding`,
                - `stride`;
            - when nn.Linear is compressed:
                - `in_features` - input features,
                - `out_features` - output features.
        """
        pass
    
    def init_device(self):
        """Sets the device, where the layer's weights are stored.
        
        Sets a value for the `device` attribute.
        """
        if  isinstance(self.layer, nn.Sequential):
            self.device = self.layer[0].weight.device
        else:
            self.device = self.layer.weight.device
            
    @abc.abstractmethod
    def estimate_rank(self,
                      weight = None,
                      rank_selection = None,
                      manual_rank = -1,
                      param_reduction_rate = None,
                      vbmf_weaken_factor = None):
        """Estimates the rank of decomposition, which is used to approximate the layer's weight.  
        
        Sets a value for the `rank` attribute.

        Parameters
        ----------
        weight : torch.tensor or None
            A weight, which is approximated.
        rank_selection : {'manual', 'param_reduction', 'vbmf'}
           A method to estimate the rank of the tensor decomposition.
        manual_rank : int or iterable or None
                - if `rank_selection` != 'manual', then `manual_rank` is -1.
                - Else, `manual_rank` equals *(int)* for CP3/CP4/SVD and *(iterable)* for Tucker2.
        param_reduction_rate : float or None
                - If `rank_selection` != 'param_reduction', then `param_reduction_rate` is None.
                - Else, `param_reduction` is a reduction factor by which the number of layer's parameters decrease after the compression step.
        vbmf_weaken_factor : float or None
                - If `rank_selection` != 'vbmf', then `vbmf_weaken_factor` is None.
                - Else, `vbmf_weaken_factor` is a weakenen factor used to increase tensor rank found via EVMBF.
        """
        pass
    
    @abc.abstractmethod
    def extract_weights(self):
        """Extracts the weight that will be approximated, and bias.
        
        Returns
        -------
        weight : torch.tensor
        bias : torch.tensor or None
        """
        pass
    
    @abc.abstractmethod
    def compute_new_weights(self, weight, bias):
        """Approximates extracted weights with a factorized tensor, whose factors are reshaped into new weights.
        
        Parameters
        ----------
        weight : torch.tensor
            A layer's weight, which is approximated.
        bias : torch.tensor or None
            A layer's bias.
        
        Returns
        -------
        weights : list(torch.tensor)
            A list of weights to initialize new layers that replace the current ones.
        biases : list
            A list ``[None,...,bias]``, where the last element equals `bias`, the rest equal None.
        """
        pass
    
    @abc.abstractmethod
    def build_new_layers(self):
        """Builds new layers to replace the current ones.
        
        Builds nn.Sequential to set `new_layers` attribute. 
        
        """
        pass
    
    
    def init_new_layers(self, weights, biases):
        """Initializes new layers with computed new weights.
        
        """
        
        for j, (w, b)  in enumerate(zip(weights, biases)):
            self.new_layers[j].weight.data = w

            if b is not None:
                self.new_layers[j].bias.data = b
            else:
                self.new_layers[j].bias = None 
    