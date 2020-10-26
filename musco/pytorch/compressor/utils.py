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

                        
def get_compressed_model(model,
                         layer_names,
                         ranks,
                         decompositions,
                         layer_types,
                         rank_selection,
                         vbmf_weaken_factors=None,
                         param_reduction_rates = None,
                         return_ranks=False):
    '''Performs one-stage model compression  by replacing layers' weights with their low-rank approximations.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to compress.
    layer_names : list
        Names of all compressing nn.Conv2d and nn.Linear layers from the initial model.
    ranks : collections.defaultdict(float)
        A dictionary ``{lname : rank for lname in layer_names}``, where `rank` is *(int or iterable)* or None.
            - `rank` is None if a corresponding layer is not compressed.
            - `rank` is -1 if the kernel of the corresponding layer is approximated using automatically defined rank value.
            - `rank` is *(int or iterable)* and rank != -1 if the kernel of the corresponding layer is approximated using a manually defined rank value.
    decompositions : collections.defaultdict(str)
        A dictionary ``{lname : decomposition for lname in layer_names}``, where decomposition is a type of tensor method applied to approximate layer's weight at the compression step.
    layer_types : collections.defaultdict(type)
        A dictionary ``{lname : linfo}`` that for each layer from the initial model contains its name `lname` and  corresponding information, such as 'type', 'kernel_size', 'groups' (last two fields are defined only for nn.Conv2d).
    vbmf_weaken_factors : collections.defaultdict(float)
        A dictionary ``{lname : vbmf_wf for lname in layer_names}``, where `vbmf_wf` is a weakenen factor used to increase tensor rank found via EVMBF.
    param_reduction_rates :  collections.defaultdict(float)
        A dictionary ``{lname : param_rr for lname in layer_names}``, where `param_rr` is a reduction factor by which the number of layer's parameters decrease after the compression step.
    
    Returns
    -------
    torch.nn.Module
        Compressed model.
        
    See Also
    --------
    .compressor.Compressor
    
    '''
    compressed_model = copy.deepcopy(model)
    new_ranks = defaultdict()
    model = None

    for lname in layer_names:
        rank = ranks[lname]

        if rank is not None:
            logging.info('Decompose layer {}'.format(lname))
            subm_names = lname.strip().split('.')

            ## model before 
            #print('subm_name: {} \n'.format(subm_names))
            layer = compressed_model.__getattr__(subm_names[0])
            for s in subm_names[1:]:
                layer = layer.__getattr__(s)
                
            decomposition = decompositions[lname]
            layer_type = layer_types[lname]['type']
                            
            if vbmf_weaken_factors is not None:
                vbmf_weaken_factor = vbmf_weaken_factors[lname]
            else:
                vbmf_weaken_factor = None
                
                
            if param_reduction_rates is not None:
                param_reduction_rate = param_reduction_rates[lname]
            else:
                param_reduction_rate = None
                
                
            rank_kwargs = Namespace(rank_selection = rank_selection,
                                    manual_rank = (rank if rank_selection == 'manual' else -1),
                                    param_reduction_rate = param_reduction_rate,
                                    vbmf_weaken_factor=vbmf_weaken_factor)
            rank_kwargs = vars(rank_kwargs)
            
            
            print(lname, decomposition)
            print(rank_kwargs)
                
            if decomposition == 'tucker2':
                decomposed_layer = Tucker2DecomposedLayer(layer, subm_names[-1], **rank_kwargs)

            elif decomposition == 'cp3':
                decomposed_layer = CP3DecomposedLayer(layer, subm_names[-1], **rank_kwargs)
                
            elif decomposition == 'cp4':
                decomposed_layer = CP4DecomposedLayer(layer, subm_names[-1], **rank_kwargs)
                
            elif decomposition == 'svd':
                if layer_type == nn.Conv2d:
                    decomposed_layer = SVDDecomposedConvLayer(layer, subm_names[-1], **rank_kwargs)
                    
                elif layer_type == nn.Linear:
                    decomposed_layer = SVDDecomposedLayer(layer, subm_names[-1], **rank_kwargs)
                    
            new_ranks[lname] = decomposed_layer.rank
            logging.info('\t new rank: {}'.format(new_ranks[lname]))

            if len(subm_names) > 1:
                m = compressed_model.__getattr__(subm_names[0])
                for s in subm_names[1:-1]:
                    m = m.__getattr__(s)
                m.__setattr__(subm_names[-1], decomposed_layer.new_layers)
            else:
                compressed_model.__setattr__(subm_names[-1], decomposed_layer.new_layers)
        else:
            logging.info('Skip layer {}'.format(lname))

    if return_ranks:
        return compressed_model, new_ranks
    else:
        return compressed_model
