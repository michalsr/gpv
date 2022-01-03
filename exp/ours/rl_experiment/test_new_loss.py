import collections
from typing import List, Optional

import numpy as np
import torch
import torchvision.ops
from allennlp.common import FromParams, Registrable, Params
from dataclasses import dataclass, asdict
from torch import nn

from exp.ours.data.dataset import Task
from exp.ours.util import our_utils
from utils.matcher import HungarianMatcher
from utils.set_criterion import SetCriterion
from torch.nn import functional as F

# range_mapping_x = {0:[0.0,.50],1:[0.5,1.],2:[0.0,0.5],3:[0.5,1.0]}
# range_mapping_y = {0:[0.0,.5],1:[0.5,1],2:[]}
# x_1,y_1,x_2,y_2 = box 
# t_x_1,t_x_2 = range_mapping_x[target]
# t_y_1,t_y_2 = range_mapping[target]
# if t_x_1 <= x_1 and t_x_2>=x_2 and t_y_1<=y_1 and t_y_2>=y_2:
#     logits = torch.zeros(4)
#     logits[target] = 1
def new_loss(bbox,target):
    target_map = {0:[0,0,.5,.5],1:[.5,0,1,0.5],2:[0,0.5,.5,1],3:[.5,.5,1,1]}
    targets = torch.tensor(target_map[target])
    print(targets,'targets')
    diff = bbox-targets 
    print(diff,'diff')
    mul = torch.tensor([1,1,-1,-1])
    prd = diff*mul 
    print(prd,'prd')
    almost_loss = torch.min(prd,torch.tensor([0,0,0,0]))
    
    subtract_one = torch.tensor([1,1,1,1])-almost_loss 
    final = torch.max(subtract_one,torch.tensor([1,1,1,1]))
    print(final)
new_loss(torch.tensor([.1,.2,.3,.4]),0)