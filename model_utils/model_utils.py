import numpy as np

def get_layer_names(model):
    layer_names = []
    conv_layer_mask = []
    
    def get_layer_name(module_name, module):
            if not '__getitem__' in dir(module):
                if module._get_name() == 'Conv2d':
                    layer_names.append(module_name)
                    conv_layer_mask.append(True)
                elif module._get_name() == 'Linear':
                    layer_names.append(module_name)
                    conv_layer_mask.append(False)
                else:
                    for l_name, l in module.named_children():
                        get_layer_name('.'.join([module_name, l_name]), l)
            else:
                for l_name, l in module.named_children():
                    get_layer_name('.'.join([module_name, l_name]), l)
                    

    for m_name, m in model.named_children():
        get_layer_name(m_name, m)
        
    return np.array(layer_names), np.array(conv_layer_mask)



