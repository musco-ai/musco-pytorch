import torch
from torch import nn
import copy

from .tucker2 import Tucker2DecomposedLayer
from .cp3 import CP3DecomposedLayer
from .cp4 import CP4DecomposedLayer
from .svd_layer import SVDDecomposedLayer
import numpy as np


def get_compressed_model(model, ranks=[], layer_names=[], decompositions=[],
                         pretrained=None,
                         vbmf_weaken_factor=None,
                         return_ranks=False):
    compressed_model = copy.deepcopy(model)
    new_ranks = copy.deepcopy(np.array(ranks, dtype=object))
    model = None

    for i, (rank, layer_name, decomposition) in enumerate(zip(ranks, layer_names, decompositions)):

        if rank is not None:
            print('Decompose layer', layer_name)
            subm_names = layer_name.strip().split('.')

            ## model before 
            # print('subm_name: {} \n'.format(subm_names))
            layer = compressed_model.__getattr__(subm_names[0])
            for s in subm_names[1:]:
                layer = layer.__getattr__(s)

            if decomposition == 'tucker2':
                decomposed_layer = Tucker2DecomposedLayer(layer, subm_names[-1], rank,
                                                          pretrained=pretrained,
                                                          vbmf_weaken_factor=vbmf_weaken_factor)

            elif decomposition == 'cp3':
                decomposed_layer = CP3DecomposedLayer(layer, subm_names[-1], rank,
                                                      pretrained=pretrained)
            elif decomposition == 'cp4':
                decomposed_layer = CP4DecomposedLayer(layer, subm_names[-1], rank,
                                                      pretrained=pretrained)
            elif decomposition == 'svd':
                decomposed_layer = SVDDecomposedLayer(layer, subm_names[-1], rank,
                                                      pretrained=pretrained,
                                                      vbmf_weaken_factor=vbmf_weaken_factor)
            try:
                new_ranks[i] = decomposed_layer.ranks
            except:
                new_ranks[i] = decomposed_layer.rank
            print('\t new rank: ', new_ranks[i])

            if len(subm_names) > 1:
                m = compressed_model.__getattr__(subm_names[0])
                for s in subm_names[1:-1]:
                    m = m.__getattr__(s)
                m.__setattr__(subm_names[-1], decomposed_layer.new_layers)
            else:
                compressed_model.__setattr__(subm_names[-1], decomposed_layer.new_layers)
        else:
            print('Skip layer', layer_name)

    if return_ranks:
        return compressed_model, new_ranks
    else:
        return compressed_model
