import torch
import torchvision

PWD = '/workspace/home/jgusak/musco'
PATH_TO_PRETRAINED = '{}/pretrained'.format(PWD)
SAVE_ROOT = '{}/results'.format(PWD)
# DATA_ROOT =  '{}/datasets'.format(PWD)
DATA_ROOT = '/workspace/raid/data/datasets'


def load_model(MODEL_NAME):
    if MODEL_NAME == 'vgg16_imagenet':
        from torchvision.models import vgg16
        PATH_TO_MODEL = '/workspace/raid/data/eponomarev/pretrained/imagenet/vgg16-397923af.pth'

        model = vgg16()
        model.load_state_dict(torch.load(PATH_TO_MODEL))

    elif MODEL_NAME == 'resnet50_imagenet':
        model = torchvision.models.resnet50(pretrained=True)
        
    elif MODEL_NAME == 'resnet18_imagenet':
        model = torchvision.models.resnet18(pretrained=True)
        
    elif MODEL_NAME == 'faster_rcnn_vgg16':
        PATH_TO_MODEL = '/workspace/raid/data/jgusak/faster_rcnn_vgg16_c.pth'
        model = torch.load(PATH_TO_MODEL)
        
    elif MODEL_NAME == 'faster_rcnn_resnet50':
#         PATH_TO_MODEL = '/workspace/raid/data/jgusak/faster_rcnn_resnet50_c.pth'
        PATH_TO_MODEL = "/workspace/home/eponomarev/tensor_od_camera_ready/demo/resnet-c4.bb"
        model = torch.load(PATH_TO_MODEL)
        
    elif MODEL_NAME == 'faster_rcnn_resnet50_fpn':
        PATH_TO_MODEL = '/workspace/raid/data/jgusak/faster_rcnn_resnet50_fpn.pth'
        model = torch.load(PATH_TO_MODEL)
               

    return model
