import torch
import torch.nn as nn

import numpy as np
from collections import defaultdict
from functools import partial

from .compute_layer_flops import *


class FlopCo():
    
    def __init__(self, model, img_size = (1, 3, 224, 224), device = 'cuda'):
        self.device = device
        self.model = model

        self.model.eval()
        self.model.to(device)

        self.img_size = img_size

        self.input_shapes = None
        self.output_shapes = None
        
        self.flops = None
        self.macs = None

        self.instances  = [nn.Conv2d,
                           nn.Linear
                          ]
        
        self.get_stats(shapes = True, flops = True, macs = True)
        
        self.total_flops = sum([sum(v) for v in self.flops.values()])
        self.total_macs = sum([sum(v) for v in self.macs.values()])
        


        
    def _save_shapes(self, name, mod, inp, out):
        self.input_shapes[name].append(inp[0].shape)
        self.output_shapes[name].append(out.shape)
        
    
    def _save_flops(self, name, mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            flops = compute_conv_flops(mod, inp[0].shape)
            
        elif isinstance(mod, nn.Linear):
            flops = compute_fc_flops(mod)

        else:
            flops = -1
        
        self.flops[name].append(flops)
        
        
    def _save_macs(self, name, mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            macs = compute_conv_flops(mod, inp[0].shape, macs = True)
            
        elif isinstance(mod, nn.Linear):
            macs = compute_fc_flops(mod, macs = True)

        else:
            macs = -1
        
        self.macs[name].append(macs)



    def get_stats(self, shapes = True, flops = False, macs = False):
        if shapes:
            self.input_shapes = defaultdict(list)
            self.output_shapes = defaultdict(list)
       
        if flops:
            self.flops = defaultdict(list)
        
        if macs:
            self.macs = defaultdict(list)

        with torch.no_grad():
            for name, m in self.model.named_modules():
                to_compute = sum(map(lambda inst : isinstance(m, inst),
                                            self.instances))
                if to_compute:

                    if shapes:
                        m.register_forward_hook(partial(self._save_shapes, name))
                    
                    if flops:
                        m.register_forward_hook(partial(self._save_flops, name))
                    
                    if macs:
                        m.register_forward_hook(partial(self._save_macs, name))


            batch = torch.rand(*self.img_size)
            if self.device == 'cuda':
                batch = batch.cuda()

            _ = self.model(batch)

            batch = None

            for name, m in self.model.named_modules():
                m._forward_pre_hooks.clear()
                m._forward_hooks.clear()
