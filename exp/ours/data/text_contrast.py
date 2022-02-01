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
class TextContrastExample:
  """
  Contrast grouping: refers to which group of image contrast examples (in each group there is one target image and multiple reference images)
  is_target: is target image 
  batches consist of contrast groups 
  initially there will be one contrast group per batch
  """

  gpv_id: str
  image_id: Union[int, str]
  answer: str
  query: str 
  contrast_group: str

  rel_query: str



  @property
  def task(self):
    return Task.TEXTCONTRAST

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("text-contrast")
class TextContrastDataset(Dataset):

  def __init__(self, split: str,raw_instances=None):
 
    

    self.split = split
    self.raw_instances = raw_instances



  def get_task(self) -> Task:
    return Task.TEXTCONTRAST

  def load(self) -> List[TextContrastExample]:
    instances = load_text_contrast(self.split,self.raw_instances)
    
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

def load_text_contrast(split=None,raw_instances=None):
  #file = join(file_paths.WEBQA_DIR, split + "_image_info.json")
  #file = file_paths.IMAGECONTRAST_DIR+'/train_large_2.json'
  #file = '/data/michal5/gpv/text_contrast/train_large.json'
  #file = '/data/michal5/gpv/lessons/text_contrast_train_10_per_group.json'
  if raw_instances == None:
    file = '/data/michal5/gpv/lessons/text_contrast_train_15_per_group.json'
    logging.info(f"Loading text contrast data from {file}")
    raw_instances = load_json_object(file)
  else:
    raw_instances = raw_instances 
  out = []
  for i, x in enumerate(raw_instances):


    if isinstance(x["image"], dict):
      image_id = x["image"]["image_id"]
    else:
      image_id = x["image"]

    ex = TextContrastExample(gpv_id=x['gpv_id'],image_id=image_id,answer=x['answer'],
    query=x['query'],contrast_group=x['contrast_group'],rel_query=x['rel_query']
      )
    out.append(ex)
    
  return out

