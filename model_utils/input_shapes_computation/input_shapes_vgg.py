import torch.nn as nn

def get_input_shapes_vgg(model, x):
    input_shapes_dict = {}
    
    for mname, m in model.features.named_children():
        if isinstance(m, nn.Sequential):
            for lname, l in m.named_children():
                input_shapes_dict['features.{}.{}'.format(mname, lname)] = x.shape
                x = l(x)
        else:
            input_shapes_dict['features.{}'.format(mname)] = x.shape
            x = m(x)
                
    input_shapes_dict['avgpool'] = x.shape
    m = nn.AdaptiveAvgPool2d((7, 7))
    x = m(x)

    x = x.view(x.size(0), -1)

    for mname, m in model.classifier.named_children():
        if isinstance(m, nn.Sequential):
            for lname, l in m.named_children():
                input_shapes_dict['classifier.{}.{}'.format(mname, lname)] = x.shape
                x = l(x)
        else:
            input_shapes_dict['classifier.{}'.format(mname)] = x.shape
            x = m(x)
    return input_shapes_dict



def get_input_shapes_faster_rcnn_vgg(model, x):
    input_shapes_dict = {}
    
    for mname, m in model.named_children():
        if isinstance(m, nn.Sequential):
            for lname, l in m.named_children():
                input_shapes_dict['{}.{}'.format(mname, lname)] = x.shape
                x = l(x)
        else:
            input_shapes_dict['{}'.format(mname)] = x.shape
            x = m(x)
                
    return input_shapes_dict
