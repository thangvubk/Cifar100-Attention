from model.fpn import *
from model.resnet import *

def get_model(opt):
    name = opt['name']
    attention_type = opt['att_type']
    if name == 'resnet':
        return resnet18(attention_type=attention_type)
    elif name == 'fishnet':
        return fishnet18(attention_type=attention_type)
    else:
        raise Exception('Unknown model ', name)

