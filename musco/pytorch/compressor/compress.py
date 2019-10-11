import numpy as np
import torch
from torch import nn
import copy
from collections import defaultdict

from absl import logging

from musco.pytorch.compressor.decompositions.tucker2 import Tucker2DecomposedLayer
from musco.pytorch.compressor.decompositions.cp3 import CP3DecomposedLayer
from musco.pytorch.compressor.decompositions.cp4 import CP4DecomposedLayer
from musco.pytorch.compressor.decompositions.svd_layer import SVDDecomposedLayer, SVDDecomposedConvLayer
                        
def get_compressed_model(model,
                         layer_names,
                         ranks,
                         decompositions,
                         layer_types,
                         rank_selection,
                         vbmf_weaken_factors=None,
                         param_reduction_rates = None,
                         pretrained=None,
                         return_ranks=False):
    '''
    layer_names:list,
    ranks: defaultdict,
    decompositions: defaultdict,
    layer_types: defaultdict,
    vbmf_weaken_factors: defaultdict
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
                
            
            print(lname, decomposition)
                
            #print(layer)
            if decomposition == 'tucker2':
                decomposed_layer = Tucker2DecomposedLayer(layer,\
                                                          subm_names[-1],\
                                                          rank_selection,\
                                                          rank,\
                                                          pretrained=pretrained,\
                                                          vbmf_weaken_factor=vbmf_weaken_factor,\
                                                          param_reduction_rate = param_reduction_rate)

            elif decomposition == 'cp3':
                decomposed_layer = CP3DecomposedLayer(layer,\
                                                      subm_names[-1],\
                                                      rank_selection,\
                                                      rank,\
                                                      pretrained=pretrained,\
                                                      param_reduction_rate = param_reduction_rate)
            elif decomposition == 'cp4':
                decomposed_layer = CP4DecomposedLayer(layer,\
                                                      subm_names[-1],\
                                                      rank_selection,\
                                                      rank,\
                                                      pretrained=pretrained,\
                                                      param_reduction_rate = param_reduction_rate)
            elif decomposition == 'svd':
                if layer_type == nn.Conv2d:
                    decomposed_layer = SVDDecomposedConvLayer(layer,\
                                                              subm_names[-1],\
                                                              rank_selection,\
                                                              rank,\
                                                              pretrained=pretrained,\
                                                              vbmf_weaken_factor=vbmf_weaken_factor,\
                                                              param_reduction_rate = param_reduction_rate)
                elif layer_type == nn.Linear:
                    decomposed_layer = SVDDecomposedLayer(layer,\
                                                          subm_names[-1],\
                                                          rank_selection,\
                                                          rank,\
                                                          pretrained=pretrained,\
                                                          vbmf_weaken_factor=vbmf_weaken_factor,\
                                                          param_reduction_rate = param_reduction_rate)
            try:
                new_ranks[lname] = decomposed_layer.ranks
            except:
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
