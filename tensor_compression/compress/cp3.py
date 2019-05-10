import numpy as np
import torch
from torch import nn

from sktensor import dtensor, ktensor, cp_als
import scipy

from .rank_selection import estimate_rank_for_compression_rate
from .cpd import recompress_ncpd_tensor


class CP3DecomposedLayer():
    def __init__(self, layer, layer_name, rank, pretrained = None):
        self.rank = rank
        self.layer_name = layer_name
        self.layer = layer
        self.pretrained = pretrained
        
        
        if  '__getitem__' in dir(self.layer):
            self.cin = self.layer[0].in_channels
            self.cout = self.layer[-1].out_channels
            
            self.kernel_size = self.layer[1].kernel_size 
            self.padding = self.layer[1].padding
            self.stride = self.layer[1].stride
           
        else:
            if self.layer._get_name() != 'Conv2d':
                raise AttributeError('only convolution layer can be decomposed')
            self.cin = self.layer.in_channels
            self.cout = self.layer.out_channels
            
            self.kernel_size = self.layer.kernel_size
            self.padding = self.layer.padding
            self.stride = self.layer.stride
                
        self.weight, self.bias = self.get_weights_to_decompose()
        
        if rank < 0:
            rank = -rank
            if  '__getitem__' in dir(self.layer):
                self.rank = estimate_rank_for_compression_rate((self.layer[1].out_channels,
                                                                 self.layer[1].in_channels,
                                                                 *self.kernel_size),
                                                                rate = rank,
                                                                key = 'cp3')
            else:
                self.rank = estimate_rank_for_compression_rate((self.cout, self.cin, *self.kernel_size),
                                            rate = rank,
                                            key = 'cp3')
        ##### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        weights, biases = self.get_cp_factors()        
        
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
        layers.append(nn.Conv2d(in_channels=self.cin,
                                    out_channels=self.rank,
                                    kernel_size = (1, 1)))
        
        layers.append(nn.Conv2d(in_channels = self.rank, 
                                    out_channels=self.rank,
                                    kernel_size = self.kernel_size,
                                    groups = int(self.rank), 
                                    padding = self.padding,
                                    stride = self.stride))
       
        layers.append(nn.Conv2d(in_channels = self.rank,
                                    out_channels = self.cout, 
                                    kernel_size = (1, 1)))
        
        return layers
    
    
    def get_weights_to_decompose(self):
        if '__getitem__' in dir(self.layer):
            w_cin = self.layer[0].weight.data
            w_z = self.layer[1].weight.data
            w_cout = self.layer[-1].weight.data

            try:
                bias = self.layer[-1].bias.data
            except:
                bias = None

            f_z = torch.transpose(w_z.reshape(w_z.shape[0], np.prod(w_z.shape[2:])), 1, 0)
            f_cin = torch.transpose(w_cin.reshape(*w_cin.shape[:2]), 1, 0)
            f_cout = w_cout.reshape(*w_cout.shape[:2])

#             weight = ktensor([dtensor(f_cout), dtensor(f_cin), dtensor(f_z)])
#             weight = torch.tensor(weight.totensor())
            weight = [f_cout, f_cin, f_z]

        else:
            weight = self.layer.weight.data
            weight = weight.reshape(*weight.shape[:2], -1)
            try:
                bias = self.layer.bias.data
            except:
                bias = None
        return weight, bias
    
    def get_cp_factors(self):
       
        if self.pretrained is not None:
            mat_dict = scipy.io.loadmat(self.pretrained)
            
            if mat_dict['R'][0][0] != self.rank:
                print('WRONG FACTORS, do not correspond to desired rank')
                
            PU_z, PU_cout, PU_cin = [Ui[0] for Ui in mat_dict['P_bals_epc_U']]
            Plmbda = mat_dict['P_bals_epc_lambda'].ravel()
            
            f_cin = np.array(PU_cin) 
            f_cout = np.array(PU_cout)
            f_z = (np.array(PU_z)*(Plmbda))
            
        else:
            if '__getitem__' in dir(self.layer):
                f_cout_old, f_cin_old, f_z_old = self.weight

                f_cout_old = np.array(f_cout_old)
                f_cin_old = np.array(f_cin_old)
                f_z_old = np.array(f_z_old)

                bias = self.bias

                f_cout, f_cin, f_z = recompress_ncpd_tensor([f_cout_old, f_cin_old, f_z_old],
                                                                 new_rank = self.rank,
                                                                 max_cycle = 500, 
                                                                 return_fit = False,
                                                                 tensor_format = 'cpd')
            else:
                if self.weight.is_cuda:
                    self.weight = self.weight.cpu()
                    if self.bias is not None:
                        self.bias = self.bias.cpu()
                weights = dtensor(self.weight)
                bias = self.bias

                T = dtensor(self.weight)
                P, fit, itr, exectimes = cp_als(T, self.rank, init='random')

                f_cout = np.array(P.U[0]) 
                f_cin = np.array(P.U[1])
                f_z = (np.array(P.U[2])*(P.lmbda))
            

        f_cin = torch.FloatTensor(np.reshape(f_cin.T, [self.rank, self.cin, 1, 1])).contiguous()
        f_z = torch.FloatTensor(np.reshape(f_z.T, [self.rank, 1, *self.kernel_size])).contiguous()
        f_cout = torch.FloatTensor(np.reshape(f_cout, [self.cout, self.rank, 1, 1])).contiguous()

        return [f_cin, f_z,  f_cout], [None, None,  bias]
    
