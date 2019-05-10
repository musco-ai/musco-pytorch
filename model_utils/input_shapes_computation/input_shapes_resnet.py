import torch.nn as nn
import torch.nn.functional as F


def get_input_shapes_resnet_bottleneck(self, x):
    input_shapes_dict = {}
    
    identity = x

    input_shapes_dict['conv1'] = x.shape
    out = self.conv1(x)
    out = self.bn1(out)
    try:
        out = self.relu(out)
    except:
        out = F.relu(out)
        
    if isinstance(self.conv2, nn.Conv2d):
        input_shapes_dict['conv2'] = out.shape
        out = self.conv2(out)
    else:
        for lname, l in self.conv2.named_children():
            input_shapes_dict['conv2.{}'.format(lname)] = out.shape
            out = l(out)
        
    out = self.bn2(out)
    try:
        out = self.relu(out)
    except:
        out = F.relu(out)

    input_shapes_dict['conv3'] = out.shape
    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        input_shapes_dict['downsample.0'] = x.shape
        identity = self.downsample(x)

    out += identity
    try:
        out = self.relu(out)
    except:
        out = F.relu(out)
    
    return out, input_shapes_dict


def get_input_shapes_resnet_basicblock(self, x):
    input_shapes_dict = {}
    
    identity = x

    input_shapes_dict['conv1'] = x.shape
    
    out = x
    if isinstance(self.conv1, nn.Conv2d):
        input_shapes_dict['conv1'] = out.shape
        out = self.conv1(out)
    else:
        for lname, l in self.conv1.named_children():
            input_shapes_dict['conv1.{}'.format(lname)] = out.shape
            out = l(out)
    out = self.bn1(out)

    try:
        out = self.relu(out)
    except:
        out = F.relu(out)

    if isinstance(self.conv2, nn.Conv2d):
        input_shapes_dict['conv2'] = out.shape
        out = self.conv2(out)
    else:
        for lname, l in self.conv2.named_children():
            input_shapes_dict['conv2.{}'.format(lname)] = out.shape
            out = l(out)
            
    out = self.bn2(out)

    if self.downsample is not None:
        input_shapes_dict['downsample.0'] = x.shape
        identity = self.downsample(x)

    out += identity
    try:
        out = self.relu(out)
    except:
        out = F.relu(out)
    
    return out, input_shapes_dict


def get_input_shapes_resnet(self, x, resnet_id = 50):
    input_shapes_dict = {}
    
    input_shapes_dict['conv1'] = x.shape
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    
    for layername, layer in zip(['layer1', 'layer2', 'layer3', 'layer4'],
                                [self.layer1, self.layer2, self.layer3, self.layer4]):
        
        for bname, block in layer.named_children():
            if resnet_id == 50:
                x, block_input_shapes_dict = get_input_shapes_resnet_bottleneck(block, x)
            elif resnet_id == 18:
                x, block_input_shapes_dict = get_input_shapes_resnet_basicblock(block, x)
            for (lname, linput_shape) in block_input_shapes_dict.items():
                name = '.'.join([layername, bname, lname])
                input_shapes_dict[name] = linput_shape

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    
    if isinstance(self.fc, nn.Linear):
        input_shapes_dict['fc'] = x.shape
        x = self.fc(x)
    else:
        for lname, l in self.fc.named_children():
            input_shapes_dict['fc.{}'.format(lname)] = x.shape
            x = l(x)
    
    return input_shapes_dict


def get_input_shapes_faster_rcnn_resnet50(self, x):
    self = self.body
    
    input_shapes_dict = {}
    
    input_shapes_dict['body.stem.conv1'] = x.shape
    x = self.stem.conv1(x)
    x = self.stem.bn1(x)
#     x = self.body.stem.relu(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    for layername, layer in zip(['body.layer1', 'body.layer2', 'body.layer3'],
                                [self.layer1, self.layer2, self.layer3]):
        
        for bname, block in layer.named_children():
            x, block_input_shapes_dict = get_input_shapes_resnet_bottleneck(block, x)
            
            for (lname, linput_shape) in block_input_shapes_dict.items():
                name = '.'.join([layername, bname, lname])
                input_shapes_dict[name] = linput_shape

    
    return input_shapes_dict


def get_input_shapes_faster_rcnn_resnet50_fpn(self, x):
    
    input_shapes_dict = {}
    
    input_shapes_dict['body.stem.conv1'] = x.shape
    x = self.body.stem.conv1(x)
    x = self.body.stem.bn1(x)
#     x = self.body.stem.relu(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    outputs = []    
    for layername, layer in zip(['body.layer1', 'body.layer2', 'body.layer3', 'body.layer4'],
                                [self.body.layer1, self.body.layer2, self.body.layer3, self.body.layer4]):
        
        for bname, block in layer.named_children():
            x, block_input_shapes_dict = get_input_shapes_resnet_bottleneck(block, x)
            
            for (lname, linput_shape) in block_input_shapes_dict.items():
                name = '.'.join([layername, bname, lname])
                input_shapes_dict[name] = linput_shape
        outputs.append(x)
        
                
    for el in outputs:
        print(el.shape)
    results = []
    
    input_shapes_dict['fpn.fpn_inner4'] = outputs[-1].shape
    x = self.fpn.fpn_inner4(outputs[-1])
    
    if isinstance(self.fpn.fpn_layer4, nn.Conv2d):
        input_shapes_dict['fpn.fpn_layer4'] = x.shape
        y = self.fpn.fpn_layer4(x)
    else:
        y = x
        for lname, l in self.fpn.fpn_layer4.named_children():
            input_shapes_dict['fpn.fpn_layer4.{}'.format(lname)] = y.shape
            y = l(y)
    
    results.append(y)
    
    for feature, inner_block, layer_block, block_id in zip(outputs[:-1][::-1],
                                                            [self.fpn.fpn_inner3, self.fpn.fpn_inner2, self.fpn.fpn_inner1],
                                                            [self.fpn.fpn_layer3, self.fpn.fpn_layer2, self.fpn.fpn_layer1],
                                                            [3, 2, 1]):
        x_top_down = F.interpolate(x, scale_factor=2, mode="nearest")
        
        input_shapes_dict['fpn.fpn_inner{}'.format(block_id)] = feature.shape
        x_lateral = inner_block(feature)
        
        x = x_lateral + x_top_down
        
        if isinstance(layer_block, nn.Conv2d):
            input_shapes_dict['fpn.fpn_layer{}'.format(block_id)] = x.shape
            y = layer_block(x)
        else:
            y = x
            for lname, l in layer_block.named_children():
                input_shapes_dict['fpn.fpn_layer{}.{}'.format(block_id, lname)] = y.shape
                y = l(y)
        
        results.insert(0, y)
     
    y = F.max_pool2d(results[-1], 1, 2, 0)
    results.extend([y])
    
    return input_shapes_dict
