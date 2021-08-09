from typing import List

from exp.ours.data.dataset import Dataset
from exp.ours.data.gpv_data import Task, GPVExample
from exp.ours.data.source_data import WebQaExample, load_instances
import numpy as np


@Dataset.register("webqa")
class Web80QaDataset(Dataset):

  def __init__(self, sample, split: str):  # Other hyper-parameters can be added here
    self.sample = sample
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    self.sample = sample
    self.split = split

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
