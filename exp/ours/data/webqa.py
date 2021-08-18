import json
import logging
from os.path import join, exists
from typing import List

from exp.ours import file_paths
from exp.ours.data.dataset import Dataset, Task
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


@Dataset.register("webqa")
class WebQaDataset(Dataset):

  def __init__(self, name, split: str, sample=None, question_types="all"):  # Other hyper-parameters can be added here
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    if name not in {"80", "fifth", "all"}:
      raise ValueError(name)
    self.sample = sample
    self.split = split
    self.name = name
    self.question_types = question_types

  def get_name(self) -> str:
    name = f"{self.name}-{self.split}"
    if self.question_types != "all":
      if self.question_types == "noun-cls":
        name = "-nouncls"
      else:
        raise NotImplementedError()
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_answer_options(self, synonyms=False):
    if synonyms:
      raise NotImplementedError()
    return WebQaAnswers(self.name, self.question_types)

  def get_task(self) -> Task:
    return Task.WEBQA

  def load(self) -> List[GPVExample]:
    instances = load_webqa(self.name, self.split)
    if self.question_types == "noun-cls":
      instances = [x for x in instances if x.meta["question_type"] == "noun_classification"]
    elif self.question_types != "all":
      raise NotImplementedError()

    if self.sample:
      instances.sort(key=lambda x: x.id)
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
    q = GPVExample(
      f"web-{x['id']}", Task.WEBQA, x["image"]["image_id"],
      query=x["query"], target_answer=x["answer"], meta=dict(question_type=x["question_type"]))
    out.append(q)
  return out
