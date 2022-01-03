import logging
import sys
from typing import Union, Optional, Dict, Any, List

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
class MILExample:
  """
  Consists of positive and negative examples for different classes 
  """

  gpv_id: str
  image_id: Union[int, str]
  answer: str
  query: str  
  correct_answer: str
  rel_query: str



  @property
  def task(self):
    return Task.MIL

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("mil")
class MILDataset(Dataset):

  def __init__(self, split: str,):
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    

    self.split = split



  def get_task(self) -> Task:
    return Task.MIL

  def load(self) -> List[MILExample]:
    instances = load_mil(self.split)
    
    return instances


def _intern(x):
  if x is None:
    return None
  return sys.intern(x)

def generate_id():
  while LAST_ID in ID_LIST:
    LAST_ID += 1
  ID_LIST.add(LAST_ID)
  return LAST_ID 

def load_mil(split):
  #file = join(file_paths.WEBQA_DIR, split + "_image_info.json")
  #file = file_paths.IMAGECONTRAST_DIR+'/train_large_2.json'
  #file = '/data/michal5/gpv/text_contrast/train_large.json'
  file = '/data/michal5/gpv/mil/train_large.json'
  logging.info(f"Loading mil data from {file}")
  raw_instances = load_json_object(file)
  out = []
  for i, x in enumerate(raw_instances):


    if isinstance(x["image"], dict):
      image_id = x["image"]["image_id"]
    else:
      image_id = x["image"]

    ex = MILExample(gpv_id=x['gpv_id'],image_id=image_id,answer=x['answer'],
      query=x['query'],correct_answer=x['correct'],rel_query=x['rel_query']
        )
    out.append(ex)
    
  return out

