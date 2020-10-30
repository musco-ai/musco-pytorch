import numpy as np
import torch
from torch import nn
import copy
from collections import defaultdict
from argparse import Namespace

from absl import logging

from .layers.tucker2 import Tucker2DecomposedLayer
from .layers.cp3 import CP3DecomposedLayer
from .layers.cp4 import CP4DecomposedLayer
from .layers.svd_layer import SVDDecomposedLayer, SVDDecomposedConvLayer
from .layers.base import DecomposedLayer
                        

def get_compressed_model(model,
                         layer_names,
                         model_stats,
                         return_ranks=False,
                         all_algo_kwargs=None,
                         model_compr_kwargs = None):
    '''Performs one-stage model compression  by replacing layers' weights with their low-rank approximations.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to compress.
    layer_names : list
        Names of all compressing nn.Conv2d and nn.Linear layers from the initial model.
    model_stats : collections.defaultdict(dict)
        A dictionary ``{lname : linfo}`` that for each layer from the initial model contains its name `lname` and  corresponding information, such as 'type', 'kernel_size', 'groups' (last two fields are defined only for nn.Conv2d).
    all_algo_kwargs :  collections.defaultdict(dict)
        A dictionary ``{decomposition : algo_kwargs}``, where `decomposition` states for the approximation type and `algo_kwargs` is a dictionary containing parameters for the approximation algorithm. For the available list of algorithm  parameters,
            - see ``tensorly.decomposition.parafac()`` arguments, if `decomposition` takes values from {'cp3', 'cp4'};
            - see ``sktensor.tucker.hooi()`` arguments, if `decomposition` is 'tucker2';
            - see ``np.linalg.svd()`` arguments, if `decomposition` is 'svd';
            - see ``tensorly.decomposition.quantized_parafac()`` arguments, if `decomposition` takes values from {'qcp3', 'qcp4'};
            
    model_compr_kwargs : collections.defaultdict(dict)
        A dictionary ``{lname : layer_compr_kwargs}`` that maps each layer in the initial model to a dictionary of parameters, which define a compression schedule for the layer.
            - If the layer is not compressing, `layer_compr_kwargs` is None.
            - Else, `layer_compr_kwargs` is a dictionary with keyword arguments defining a layer compression schedule. 


    Returns
    -------
    torch.nn.Module
        Compressed model.
        
    See Also
    --------
    musco.pytorch.compressor.compressor.Compressor
    
    '''
    compressed_model = copy.deepcopy(model)
    new_ranks = defaultdict()
    model = None
    
    for lname in layer_names:
        compr_kwargs = model_compr_kwargs[lname]
        
        if compr_kwargs is None:
            logging.info('Skip layer {}'.format(lname))
            continue
        else:
            logging.info('Compress layer {}'.format(lname))
            subm_names = lname.strip().split('.')
            
            layer = compressed_model.__getattr__(subm_names[0])
            for s in subm_names[1:]:
                layer = layer.__getattr__(s)
                
            decomposition = compr_kwargs['decomposition']
            layer_type = model_stats.ltypes[lname]['type']
            
            algo_kwargs = all_algo_kwargs[decomposition]
            
            print(lname, compr_kwargs)
            
            if decomposition == 'tucker2':
                decomposed_layer = Tucker2DecomposedLayer(layer, subm_names[-1], algo_kwargs, **compr_kwargs)

            elif decomposition in ['cp3', 'qcp3']:
                decomposed_layer = CP3DecomposedLayer(layer, subm_names[-1], algo_kwargs, **compr_kwargs)
                
            elif decomposition in ['cp4', 'qcp4']:
                decomposed_layer = CP4DecomposedLayer(layer, subm_names[-1], algo_kwargs, **compr_kwargs)
                
            elif decomposition == 'svd':
                if layer_type == nn.Conv2d:
                    decomposed_layer = SVDDecomposedConvLayer(layer, subm_names[-1], algo_kwargs, **compr_kwargs)
                    
                elif layer_type == nn.Linear:
                    decomposed_layer = SVDDecomposedLayer(layer, subm_names[-1], algo_kwargs, **compr_kwargs)
                    
            new_ranks[lname] = decomposed_layer.rank
            logging.info('\t new rank: {}'.format(new_ranks[lname]))

            if len(subm_names) > 1:
                m = compressed_model.__getattr__(subm_names[0])
                for s in subm_names[1:-1]:
                    m = m.__getattr__(s)
                m.__setattr__(subm_names[-1], decomposed_layer)
            else:
                compressed_model.__setattr__(subm_names[-1], decomposed_layer)
                
            model_compr_kwargs[lname]['curr_compr_iter'] += 1
            
    if return_ranks:
        return compressed_model, new_ranks
    else:
        return compressed_model


def standardize_model(model):
    """Replace custom layers with standard nn.Module layers.
    
    Relplace each layer of type DecomposedLayer with nn.Sequential.
    """

    for mname, m in model.named_modules():
        if isinstance(m, DecomposedLayer):
            subm_names = mname.strip().split('.')

            if len(subm_names) > 1:
                m = model.__getattr__(subm_names[0])
                for s in subm_names[1:-1]:
                    m = m.__getattr__(s)
                m.__setattr__(subm_names[-1], m.__getattr__(subm_names[-1]).new_layers)
            else:
                model.__setattr__(subm_names[-1], m.new_layers)