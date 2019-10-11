import torch.nn as nn
from collections import defaultdict
import copy

from musco.pytorch.compressor.compress import get_compressed_model



class Compressor():
    """
    Class to perform automated compression using VBMF rank selection.
    Layers of type nn.Conv2d 1x1 and nn.Linear are decomposed using SVD, nn.Conv2d kxk(k>1) - using Tucker2 decomposition.
    """
    def __init__(self, model,\
                 model_stats,\
                 rank_selection = None,\
                 conv2d_nn_decomposition = None,\
                 ranks = None,\
                 vbmf_weakenen_factors = None,\
                 param_reduction_rates = None,\
                 ft_every = None,\
                 nglobal_compress_iters = 1):
        """
        Parameters
        -----------
        model: torch.nn.Module, model to compress
        model_stats: FlopCo,
        rank_selection: str, ['vbmf', 'param_reduction', 'manual']
            rank selection method to decompose nn.Conv2d kernel of spacial size nxn, n>1 
        
        conv2d_nn_decomposition: str, ['tucker2'/'cp3'/'cp4']
        
        vbmf_weakenen_factors: collections.defaultdict(float), dictionary {key : value},
        where key - layer name, value - weakenen factor used to increase tensor rank found via VMBF 
        
        param_reduction_rates: collections.defaultdict(float), dictionary {key : value}, 
        where key - layer name, value - reduction factor by which we decrease number of layer parameters  
        """
        
        self.ltypes = model_stats.ltypes
        self.lnames = [lname for lname in self.ltypes.keys() if\
                       self.ltypes[lname]['type'] in [nn.Conv2d, nn.Linear]]
        
        self.rank_selection = rank_selection
        self.conv2d_nn_decomposition = conv2d_nn_decomposition
        
        self.ranks = None
        self.initialize_ranks(ranks)
        
        self.pop_noncompressing_lnames()
        
        self.vbmf_wfs = None
        self.param_rrs = None
        
        if self.rank_selection == 'vbmf':
            self.initialize_vbmf_wfs(vbmf_weakenen_factors)

        elif self.rank_selection == 'param_reduction':
            self.initialize_param_rrs(param_reduction_rates)
            
        self.decompositions = None
        self.initialize_decompositions()
        
            
        self.ft_every = len(self.lnames)
        if ft_every:
            self.ft_every = min(ft_every, len(self.lnames))
        
        self.nglobal_compress_iters = nglobal_compress_iters
        self.niters = (self.nglobal_compress_iters * len(self.lnames)) // self.ft_every
        self.niters += ((self.nglobal_compress_iters * len(self.lnames)) % self.ft_every != 0)
        
        self.curr_iter = 0
        self.curr_ncompr_layers = 0
        self.done = False
        
        self.compressed_model = copy.deepcopy(model)
        
        
    def __str__(self):
        print_info = "\n".join([str({k:v}) for k,v in self.__dict__.items() if k != "ltypes"])
        
        return str(self.__class__) + ": \n" + print_info
    
    def compression_step(self):
        start = (self.curr_iter * self.ft_every) % len(self.lnames)
        end = start + self.ft_every
    
        lnames = self.lnames[start : end]
        
        self.curr_iter += 1
        
        if self.curr_iter >= self.niters:
            self.done = True
        
        if end > len(self.lnames):
            lnames += self.lnames[0 : end - len(self.lnames)]
            
        
        for i, lname in enumerate(lnames):

            self.compressed_model, new_ranks = get_compressed_model(self.compressed_model,
                                            layer_names = [lname],
                                            ranks = self.ranks,
                                            decompositions = self.decompositions,
                                            rank_selection = self.rank_selection,
                                            vbmf_weaken_factors = self.vbmf_wfs,
                                            param_reduction_rates = self.param_rrs,                        
                                            layer_types = self.ltypes,
                                            return_ranks = True)
            self.ranks.update(new_ranks)
            
            self.curr_ncompr_layers += 1
            if self.curr_ncompr_layers >= self.nglobal_compress_iters * len(self.lnames):
                break
            


    def pop_noncompressing_lnames(self):
        for lname in self.lnames:
            if self.ranks[lname] is None:
                self.ranks.pop(lname)
                
        self.lnames = list(self.ranks.keys())
                
    
    def initialize_param_rrs(self, param_rrs):
        self.param_rrs = {k : 2 for k in self.lnames}

        if param_rrs is not None:
            updates = {k:v for k,v in param_rrs.items() if k in self.lnames}

            self.param_rrs.update(updates)       
    
    
    def initialize_vbmf_wfs(self, vbmf_wfs):

        self.vbmf_wfs = {k : 0.8 for k in self.lnames}

        if vbmf_wfs is not None:
            updates = {k:v for k,v in vbmf_wfs.items() if k in self.lnames}

            self.vbmf_wfs.update(updates)



    def initialize_ranks(self, ranks):

        self.ranks = {k : 0 for k in self.lnames}

        if ranks is not None:
            updates = {k:v for k,v in ranks.items() if k in self.lnames}

            self.ranks.update(updates)
        


    def initialize_decompositions(self):
        self.decompositions = defaultdict()

        for lname in self.lnames:
            options = self.ltypes[lname]

            if options['type'] == nn.Conv2d and options['kernel_size'] != (1, 1):
                self.decompositions[lname] = self.conv2d_nn_decomposition
            else:
                self.decompositions[lname] = 'svd'
                    

                    
class CompressorVBMF(Compressor):
    def __init__(self, model,\
                 model_stats,\
                 ranks = None,\
                 vbmf_weakenen_factors = None,\
                 ft_every = None,\
                 nglobal_compress_iters = 1):
        
        
        super(CompressorVBMF, self).__init__(
             model,\
             model_stats,\
             rank_selection = 'vbmf',\
             conv2d_nn_decomposition = 'tucker2',\
             ranks = ranks,\
             vbmf_weakenen_factors = vbmf_weakenen_factors,\
             ft_every = ft_every,\
             nglobal_compress_iters = nglobal_compress_iters)
        

class CompressorPR(Compressor):
    def __init__(self, model,\
                 model_stats,\
                 conv2d_nn_decomposition = None,\
                 ranks = None,\
                 param_reduction_rates = None,\
                 ft_every = None,\
                 nglobal_compress_iters = 1):
        
        super(CompressorPR, self).__init__(
             model,\
             model_stats,\
             rank_selection = 'param_reduction',\
             conv2d_nn_decomposition = conv2d_nn_decomposition,\
             ranks = ranks,\
             param_reduction_rates = param_reduction_rates,\
             ft_every = ft_every,\
             nglobal_compress_iters = nglobal_compress_iters)
        
        
class CompressorManual(Compressor):
      def __init__(self, model,\
                 model_stats,\
                 conv2d_nn_decomposition = None,\
                 ranks = None,\
                 ft_every = None,\
                 nglobal_compress_iters = 1): 
            
        super(CompressorManual, self).__init__(
                model,\
                model_stats,\
                rank_selection = 'manual',\
                conv2d_nn_decomposition = conv2d_nn_decomposition,\
                ranks = ranks,\
                vbmf_weakenen_factors = None,\
                param_reduction_rates = None,\
                ft_every = ft_every,\
                nglobal_compress_iters = nglobal_compress_iters)
        
        
