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
@PredictionArg.register("grid-answers")
class GridCompareAnswers(PredictionArg, list):
  def __init__(self):

    for part in ["train", "test", "val"]:
      examples += GridCompareDataset(part).load()
    answers = sorted(set(x.answer for x in examples))
       
    super().__init__(answers)


# @PredictionArg.register("webqa-answer-boost")
# class WebQaAnswersBoost(MaskSpec):

#   def __init__(self, val: float, question_types="all"):
#     self.val = val
#     self.question_types = question_types
#     self.data = WebQaAnswers("all-v2", question_types)

#   def get_target_words(self):
#     return self.data

#   def target_eos(self):
#     return False


@dataclass
class GridCompareExample:
  """
  qtypes:
  q: What is the full query?
  1n: What is the noun?
  1v: What is the verb?
  1a: What is the adj?
  2a: What is the adj given the noun?
  2v: What is the verb given the noun?
  """

  gpv_id: str
  image_id: Union[int, str]
  adj_category: str
  noun_category: str 
  answer: str


  @property
  def task(self):
    return Task.GRIDCOMPARE

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("grid-compare")
class GridCompareDataset(Dataset):
 

  def __init__(self, split: str):
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    # if isinstance(qtypes, str):
    #   self.qtypes = self.QTYPES_NAME_TO_TYPES[qtypes]
    # else:
    #   assert len(qtypes) == len(set(qtypes))
    #   self.qtypes = qtypes
    # self.sample = sample
    self.split = split

  # def get_qtypes_name(self):
  #   if len(self.qtypes) == 1:
  #     return self.qtypes[0]
  #   else:
  #     return self.QTYPES_TYPES_TO_NAMES[frozenset(self.qtypes)]

  # def get_name(self) -> str:
  #   name = f"webqa-v4-{self.split}"
  #   name += f"-{self.get_qtypes_name()}"
  #   if self.sample is not None:
  #     name += f"-s{int_to_str(self.sample)}"
  #   return name

  def get_answer_options(self, synonyms=False):
    if synonyms:
      raise NotImplementedError()
    return GridCompareAnswers()

  def get_task(self) -> Task:
    return Task.GRIDCOMPARE

  def load(self) -> List[GridCompareExample]:
    instances = load_grid_compare(self.split)
    if self.sample:
      instances.sort(key=lambda x: x.gpv_id)
      np.random.RandomState(613423).shuffle(instances)
      return instances[:self.sample]
    else:
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

def load_grid_compare(split):
  file = join(file_paths.GRIDCOMPARE, split + "_image_info.json")
  
  prefix = "web" if split == "val" else f"web-{split}"
  logging.info(f"Loading grid compare data from {file}")
  raw_instances = load_json_object(file)
  out = []
  for i, x in enumerate(raw_instances):


    if isinstance(x["image"], dict):
      image_id = x["image"]["image_id"]
    else:
      image_id = x["image"]

    ex = GridCompareExample(
     image_id,None,None)
    )
    web_ids = x['web_id']
    ex_types = []
    if "1n" in qtypes:
      ex_types.append(("1n", ex.noun))
    if "q" in qtypes:
      ex_types.append(("q", _intern(x["bing_query"])))
    if ex.verb is not None:
      ex_types += [(q, ex.verb) for q in ["1v", "2v"] if q in qtypes]
    if ex.adj is not None:
      ex_types += [(q, ex.adj) for q in ["1a", "2a"] if q in qtypes]
    for q, ans in ex_types:
      out.append(replace(ex, qtype=q, answer=ans, gpv_id=f"{prefix}-{q}-{web_ids[q]}"))
  return out

