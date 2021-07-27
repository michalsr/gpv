from typing import List

from allennlp.common import Registrable

from exp.ours.data import source_data
import numpy as np

from exp.ours.data.source_data import ID_TO_COCO_CATEGORY, CocoCaptions
from exp.ours.data.gpv_data import Task
from exp.ours.util.py_utils import int_to_str


class Dataset(Registrable):

  def get_task(self) -> Task:
    raise NotImplementedError()

  def get_name(self) -> str:
    raise NotImplementedError()

  def load(self) -> List:
    raise NotImplementedError()


class InMemoryDataset(Dataset):
  def __init__(self, data, task, name):
    self.data = data
    self.task = task
    self.name = name

  def get_task(self) -> Task:
    return self.task

  def get_name(self) -> str:
    return self.name

  def load(self) -> List:
    return self.data


def split_seen_unseen(instances):
  unseen_instances = []
  seen_instances = []
  for instance in instances:
    if isinstance(instance, CocoCaptions):
      unseen = sum(len(x.meta["gpv1-unseen"]) > 0 for x in instance.captions)
      unseen = unseen > 1
    else:
      unseen = instance.meta["gpv1-unseen"]
    if unseen:
      unseen_instances.append(instance)
    else:
      seen_instances.append(instance)
  return unseen_instances, seen_instances


@Dataset.register("gpv")
class GpvDataset(Dataset):
  KINDS = {
    Task.VQA: source_data.load_gpv_vqa,
    Task.CLS: source_data.load_gpv_cls,
    Task.CLS_IN_CONTEXT: source_data.load_gpv_ident,
    Task.CAPTIONING: source_data.load_gpv_captioning,
    Task.DETECTION: source_data.load_gpv_boxes
  }

  UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass']
  UNSEEN2 = ['banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']

  UNSEEN_GROUPS = {
    Task.VQA: UNSEEN1,
    Task.CLS: UNSEEN2,
    Task.CLS_IN_CONTEXT: UNSEEN2,
    Task.CAPTIONING: UNSEEN1,
    Task.DETECTION: UNSEEN2
  }

  def __init__(self, task: Task, split: str, gpv_split=True,
               sample=None, seen_sample=None, unseen_sample=None):
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    if sample is not None and (seen_sample is not None or unseen_sample is not None):
      raise ValueError("Cannot specify sample and seen/unseen sample")
    self.sample = sample
    self.task = task
    self.split = split
    self.gpv_split = gpv_split
    self.seen_sample = seen_sample
    self.unseen_sample = unseen_sample

  def get_name(self):
    kind = "gpvsce" if self.gpv_split else "gpv"
    name = f"{kind}-{self.task}-{self.split}"
    if self.seen_sample is not None:
      name += f"-se{int_to_str(self.seen_sample)}"
    if self.unseen_sample is not None:
      name += f"-us{int_to_str(self.unseen_sample)}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_task(self) -> Task:
    return self.task

  def load(self):
    instances = self.KINDS[self.task](self.split, self.gpv_split)
    if self.seen_sample is not None or self.unseen_sample is not None:
      instances.sort(key=lambda x: x.get_gpv_id())
      np.random.RandomState(613423).shuffle(instances)
      unseen, seen = split_seen_unseen(instances)
      return unseen[:self.unseen_sample] + seen[:self.seen_sample]
    elif self.sample:
      instances.sort(key=lambda x: x.get_gpv_id())
      np.random.RandomState(613423).shuffle(instances)
      return instances[:self.sample]
    else:
      return instances
