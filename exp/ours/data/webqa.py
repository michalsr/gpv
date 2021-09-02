import json
import logging
import sys
from collections import Counter, defaultdict
from os.path import join, exists
from typing import List, Union, Optional, Dict, Any

from dataclasses import dataclass

from exp.ours import file_paths
from exp.ours.boosting import MaskSpec
from exp.ours.data.dataset import Dataset, Task, ClsExample
from exp.ours.data.gpv import load_instances
from exp.ours.data.gpv_example import GPVExample
import numpy as np

from exp.ours.models.model import PredictionArg
from exp.ours.util.py_utils import int_to_str
from utils.io import load_json_object, dump_json_object


@PredictionArg.register("webqa-answers")
class WebQaAnswers(PredictionArg, list):
  def __init__(self, name, question_types="all"):
    self.name = name
    self.question_types = question_types

    answer_set_file = join(file_paths.WEBQA_DIR, name, "answer_set.json")
    if exists(answer_set_file):
      answers = load_json_object(answer_set_file)
    else:
      cache_file = join(file_paths.CACHE_DIR, f"webqa-{name}-answers.json")
      if exists(cache_file):
        answers = load_json_object(cache_file)
      else:
        logging.info(f"Computing and caching webqa {name} answers")
        examples = []
        for part in ["train", "test", "val"]:
          examples += WebQaDataset(name, part, question_types=self.question_types).load()
        answers = sorted(set(x.target_answer for x in examples))
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


def get_webqa_answers_kinds():
  cache_file = join(file_paths.CACHE_DIR, f"webqa-answers-types.json")
  if exists(cache_file):
    return load_json_object(cache_file)
  else:
    logging.info("Computing webqa answer types...")
    data = WebQaDataset("all", "train").load()
    kind_answers = defaultdict(Counter)
    for ex in data:
      kind_answers[ex.meta["question_type"][0]][ex.target_answer] += 1

    kind_answers = {k: dict(v) for k, v in kind_answers.items()}
    dump_json_object(kind_answers, cache_file, indent=2)
    return kind_answers


@dataclass
class WebQaExample:
  gpv_id: str
  task: Task
  image_id: Union[int, str]
  query: str
  answer: str
  question_type: str
  meta: Optional[Dict[str, Any]] = None

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("webqa")
class WebQaDataset(Dataset):
  QUESTION_TYPE_GROUPS = {
    "nouncls": {"noun_classification"},
    "subset1": {"5n", "1v", "2v", "6v", "5v", "4v", "3v", "8v", "7v", "1a", "5a", "1n", "3n"},
    "n1": {"1n"},
    "noun-all": {"1n", "3n", "5n", "7n"},
    "basic": {"1n", "1a", "1v"},
    "adj-all": set(f"{i}a" for i in range(1, 9)),
    "verb-all": set(f"{i}v" for i in range(1, 8)),
    "non-noun": set([f"{i}v" for i in range(1, 8)] + [f"{i}a" for i in range(1, 9)])
  }

  def __init__(self, name, split: str, sample=None, question_types="all"):  # Other hyper-parameters can be added here
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    if name not in {"80", "fifth", "all", "all-v2"}:
      raise ValueError(name)
    self.sample = sample
    self.split = split
    self.name = name
    self.question_types = question_types

  def get_name(self) -> str:
    name = f"webqa-{self.name}-{self.split}"
    if self.question_types != "all":
      name = f"-{self.question_types}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_answer_options(self, synonyms=False):
    if synonyms:
      raise NotImplementedError()
    return WebQaAnswers(self.name, self.question_types)

  def get_task(self) -> Task:
    return Task.CLS

  def load(self) -> List[WebQaExample]:
    instances = load_webqa(self.name, self.split)
    if self.question_types != "all":
      target_kinds = self.QUESTION_TYPE_GROUPS[self.question_types]
      prev_len = len(instances)
      instances = [x for x in instances if x.question_type in target_kinds]
      logging.info(f"Selected {len(instances)} if {prev_len} questions for qtype {self.question_types}")
      assert len(instances) > 0

    if self.sample:
      instances.sort(key=lambda x: x.gpv_id)
      np.random.RandomState(613423).shuffle(instances)
      return instances[:self.sample]
    else:
      return instances


def load_webqa(part, split):
  """Load WebQA data"""
  file = join(file_paths.WEBQA_DIR, part, split + ".json")
  logging.info(f"Loading webqa data from {file}")
  raw_instances = load_json_object(file)
  out = []
  for x in raw_instances:
    q = WebQaExample(
      f"web-{x['id']}", Task.WEBQA, x["image"]["image_id"],
      x["query"],
      sys.intern(x["answer"]), x["question_type"]
    )
    out.append(q)
  return out
