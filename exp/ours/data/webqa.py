import json
from typing import List

from exp.ours import file_paths
from exp.ours.data.dataset import Dataset, Task
from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.source_data import WebQaExample, load_instances
import numpy as np

from exp.ours.models.model import PredictionArg


@PredictionArg.register("webqa-list")
class WebQa80Answers(PredictionArg, list):
  def __init__(self):
    with open(file_paths.WEBQA80_ANSWERS) as f:
      super().__init__(json.load(f))


@Dataset.register("webqa")
class Web80QaDataset(Dataset):

  def __init__(self, sample, split: str):  # Other hyper-parameters can be added here
    self.sample = sample
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    self.sample = sample
    self.split = split

  def get_answer_options(self, synonyms=False):
    if synonyms:
      raise NotImplementedError()
    return WebQa80Answers()

  def get_task(self) -> Task:
    return Task.WEBQA

  def load(self) -> List[GPVExample]:
    instances = load_web80(self.split)
    if self.sample:
      instances.sort(key=lambda x: x.id)
      np.random.RandomState(613423).shuffle(instances)
      return instances[:self.sample]
    else:
      return instances


def load_web80(split) -> List[GPVExample]:
  """Load WebQA data"""
  raw_instances = load_instances("web_80", split)
  out = []
  for x in raw_instances:
    q = GPVExample(
      f"web-{x['id']}", Task.WEBQA, x["image"]["image_id"],
      query=x["query"], target_answer=x["answer"])
    out.append(q)
  return out
