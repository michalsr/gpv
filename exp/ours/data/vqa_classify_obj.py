import logging
import sys
from typing import List, Union, Optional, Tuple, Dict, Counter, Any

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
class VQA_CLS_OBJ_Example:
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
    return Task.VQACLSOBJ

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("vqa-cls-obj")
class VQA_CLS_OBJ_Dataset(Dataset):

  def __init__(self, split: str,sample=0.2):
 
    

    self.split = split
    self.sample = sample



  def get_task(self) -> Task:
    return Task.VQACLSOBJ

  def load(self) -> List[VQA_CLS_OBJ_Example]:
    instances = load_vqa_cls_obj(self.split,self.sample)
    
    return instances


def _intern(x):
  if x is None:
    return None
  return sys.intern(x)


def load_vqa_cls_obj(split=None,sample=0.1):
  file = '/data/michal5/gpv/lessons/vqa_classify_obj_coco_all_web.json'
  logging.info(f"Loading vqa classify object data from {file}")
  raw_instances = load_json_object(file)
  out = []
  for i, x in enumerate(raw_instances):


    if isinstance(x["image"], dict):
      image_id = x["image"]["image_id"]
    else:
      image_id = x["image"]

    ex = VQA_CLS_OBJ_Example(gpv_id=x['gpv_id'],image_id=image_id,answer=x['answer'],
     question=x['query']
      )
    out.append(ex)
  #filtered = out
  filtered = np.random.choice(out,size=int(.2*len(out)),replace=False).tolist()
  return filtered

