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


@PredictionArg.register("webqa-answers")
class WebQaAnswers(PredictionArg, list):
  def __init__(self, question_types="all"):
    self.question_types = question_types

    answer_set_file = join(file_paths.WEBQA_DIR, "answer_set.json")
    if exists(answer_set_file):
      answers = load_json_object(answer_set_file)
    else:
      cache_file = join(file_paths.CACHE_DIR, f"webqa-answers.json")
      if exists(cache_file):
        answers = load_json_object(cache_file)
      else:
        logging.info(f"Computing and caching webqa answers")
        examples = []
        for part in ["train", "test", "val"]:
          examples += WebQaDataset(part, qtypes=self.question_types).load()
        answers = sorted(set(x.answer for x in examples))
        dump_json_object(answers, cache_file, indent=2)
    super().__init__(answers)


@PredictionArg.register("webqa-answer-boost")
class WebQaAnswersBoost(MaskSpec):

  def __init__(self, val: float, question_types="all"):
    self.val = val
    self.question_types = question_types
    self.data = WebQaAnswers("all-v2", question_types)

  def get_target_words(self):
    return self.data

  def target_eos(self):
    return False


@Dataset.register("webqa-notemplates")
class WebQaNoTemmplatesDataset(Dataset):
  def __init__(self, split: str, sample=None, qtypes="all"):
    self.split = split
    self.sample = sample
    self.qtypes = qtypes

  def get_name(self) -> str:
    name = f"webqa-v3"
    if isinstance(self.qtypes, tuple):
      name += f"-{''.join(sorted(self.qtypes))}"
    else:
      name += f"-{self.qtypes}"
    name += f"-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_task(self) -> Task:
    return Task.WEBQA

  def load(self) -> List[GPVExample]:
    instances = load_webqa_notemplates(self.split, self.qtypes)
    if self.sample:
      instances.sort(key=lambda x: x.gpv_id)
      np.random.RandomState(613423).shuffle(instances)
      return instances[:self.sample]
    else:
      return instances


def load_webqa_notemplates(split, qtypes) -> List[GPVExample]:
  file = join(file_paths.WEBQA_NO_TEMPLATES_DIR, f"{split}.json")
  logging.info(f"Loading webqa data from {file}")
  data = load_json_object(file)
  out = []
  for item in data:
    if qtypes == "all":
      keep = True
    if isinstance(qtypes, tuple):
      keep = item["question_type"] in qtypes
    else:
      raise NotImplementedError(qtypes)
    if keep:
      out.append(GPVExample(
        id=item["id"],
        image_id=item["image"]["image_id"],
        task=Task.WEBQA,
        query=item["query"],
        target_answer=item["answer"],
      ))
  return out


@dataclass
class WebQaExample:
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
  answer: str
  qtype: Union[str, List[str]]
  noun: Optional[str]
  verb: Optional[str]
  adj: Optional[str]
  meta: Optional[Dict[str, Any]] = None

  @property
  def task(self):
    return Task.WEBQA

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("webqa-v2")
class WebQaDataset(Dataset):
  QTYPES_NAME_TO_TYPES = {
    "1": ("1n", "1v", "1a"),
    "1and2": ("1n", "1v", "1a", "2a", "2v"),
    "1q": ("q", "1n", "1v", "1a"),
    "basic": ("q", "1n", "1v", "1a", "2a", "2v")
  }
  QTYPES_TYPES_TO_NAMES = {
    frozenset(v): k for k, v in QTYPES_NAME_TO_TYPES.items()
  }

  def __init__(self, split: str, sample=None, qtypes="basic"):
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    if isinstance(qtypes, str):
      self.qtypes = self.QTYPES_NAME_TO_TYPES[qtypes]
    else:
      assert len(qtypes) == len(set(qtypes))
      self.qtypes = qtypes
    self.sample = sample
    self.split = split

  def get_qtypes_name(self):
    if len(self.qtypes) == 1:
      return self.qtypes[0]
    else:
      return self.QTYPES_TYPES_TO_NAMES[frozenset(self.qtypes)]

  def get_name(self) -> str:
    name = f"webqa-v4"
    name += f"-{self.get_qtypes_name()}"
    name += f"-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_answer_options(self, synonyms=False):
    if synonyms:
      raise NotImplementedError()
    return WebQaAnswers(self.qtypes)

  def get_task(self) -> Task:
    return Task.WEBQA

  def load(self) -> List[WebQaExample]:
    instances = load_webqa(self.split, self.qtypes)
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


def load_webqa(split, qtypes):
  file = join(file_paths.WEBQA_DIR, split + "_image_info.json")

  prefix = "web" if split == "val" else f"web-{split}"
  logging.info(f"Loading webqa data from {file}")
  raw_instances = load_json_object(file)
  out = []
  for i, x in enumerate(raw_instances):
    if isinstance(x["image"], dict):
      image_id = x["image"]["image_id"]
    else:
      image_id = x["image"]

    ex = WebQaExample(
      None, image_id, None,
      None, noun=_intern(x["noun"]),
      adj=_intern(x["adj"]), verb=_intern(x["verb"])
    )

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
      out.append(replace(ex, qtype=q, answer=ans, gpv_id=f"{prefix}{i}-{q}"))
  return out

