from dotmap import DotMap
import torch


class CONFIG:
    nStacks = 8
    nFeatures = 256
    nModules = 1
    nJoints = 16
    nDepth = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parameter_dir = './save/SH'
    log_dir = './log/'
