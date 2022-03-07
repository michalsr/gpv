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
class SynonymExample:
  """
  Consists of positive and negative examples for different classes 
  """

  gpv_id: str
  image_id: Union[int, str]
  query: str  
  rel_query: str
  answer: str
  syn_id: int



  @property
  def task(self):
    return Task.SYNONYM

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("synonym")
class SynonymDataset(Dataset):

  def __init__(self, split: str,raw_instances=None):
    
    

    self.split = split
    self.raw_instances = raw_instances



  def get_task(self) -> Task:
    return Task.SYNONYM

  def load(self) -> List[SynonymExample]:
    instances = load_synonym(self.split,self.raw_instances)
    
    return instances


def _intern(x):
  if x is None:
    return None
  return sys.intern(x)



def load_synonym(split=None,raw_instances=None):
  if raw_instances == None:
    file = '/data/michal5/gpv/lessons/synonym_train_super_rel.json'
    logging.info(f"Loading synonym data from {file}")
    raw_instances = load_json_object(file)
  else:
    raw_instances = raw_instances
  out = []
  for i, ex in enumerate(raw_instances):
  

    if isinstance(ex["image"], dict):
      image_id = ex["image"]["image_id"]
 
    else:
      image_id = ex["image"]
   
    ex= SynonymExample(gpv_id=ex["gpv_id"],image_id=image_id,query=ex['query'],answer=ex['answer'],rel_query=ex['rel_query'],syn_id=ex['syn_id'])
    out.append(ex)
    
  return out

