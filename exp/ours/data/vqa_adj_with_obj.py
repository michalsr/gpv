import logging
import sys
from typing import List, Union, Optional, Tuple, Dict, Counter, Any
from unittest.util import strclass

from dataclasses import dataclass, replace

from exp.ours import file_paths
from exp.ours.boosting import MaskSpec
from exp.ours.data.dataset import Dataset, Task
from exp.ours.data.gpv_example import GPVExample
from exp.ours.models.model import PredictionArg
from os.path import join, exists

from exp.ours.util.py_utils import int_to_str
from utils.io import load_json_object, dump_json_object
import numpy as np

ID_LIST = set([0])
LAST_ID = 0
@dataclass
class VQA_ADJ_W_OBJ_Example:
  """
  Contrast grouping: refers to which group of image contrast examples (in each group there is one target image and multiple reference images)
  is_target: is target image 
  batches consist of contrast groups 
  initially there will be one contrast group per batch
  """

  gpv_id: str
  image_id: Union[int, str]
  question: str 
  answer: str



  @property
  def task(self):
    return Task.ADJWOBJ

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("vqa-adj-w-obj")
class VQA_ADJ_W_OBJ_Dataset(Dataset):

  def __init__(self, split: str,sample=0.2):
 
    

    self.split = split
    self.sample = sample



  def get_task(self) -> Task:
    return Task.ADJWOBJ

  def load(self) -> List[VQA_ADJ_W_OBJ_Example]:
    instances = load_adj_w_obj(self.split,self.sample)
    
    return instances


def _intern(x):
  if x is None:
    return None
  return sys.intern(x)


def load_adj_w_obj(split=None,sample=0.2):
  if split == 'all':

    file = '/data/michal5/gpv/lessons/vqa_adj_with_obj_all_web.json'
    logging.info("Using entries from all web data")
  else:
    file = '/data/michal5/gpv/lessons/vqa_adj_with_obj_coco_only.json'
  logging.info(f"Loading vqa adj with object data from {file}")
  raw_instances = load_json_object(file)
  out = []
  for i, x in enumerate(raw_instances):


    if isinstance(x["image"], dict):
      image_id = x["image"]["image_id"]
    else:
      image_id = x["image"]
  
    #print(x['answer'],'hi')
    ex = VQA_ADJ_W_OBJ_Example(gpv_id=x['gpv_id'],image_id=image_id,answer=x['answer'],
     question=x['query']
      )
    out.append(ex)
  if split == 'all':
    filtered = np.random.choice(out,size=int(0.2*len(out)),replace=False)
    filtered = filtered.tolist()
  else:
     filtered = out
  return filtered

