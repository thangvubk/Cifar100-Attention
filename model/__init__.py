from model.fpn import *
from model.resnet import *

def get_model(opt):

    name = opt['name']
    t_att = opt['t_att']
    t_norm = opt['t_norm']

    if name == 'resnet':
        return resnet18(t_att=t_att, t_norm=t_norm)
    elif name == 'fishnet':
        return fishnet18(t_att=t_att, t_norm=t_norm)
    else:
        raise Exception('Unknown model ', name)

