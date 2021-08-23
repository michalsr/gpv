import json
from typing import List

from allennlp.common import Registrable
from dataclasses import dataclass

from data.coco.synonyms import SYNONYMS
from exp.ours import file_paths
from exp.ours.data.dataset import Task
from exp.ours.data.gpv import GpvDataset, COCO_CATEGORIES
from exp.ours.data.opensce import OpenSceCategories, OPENSCE_SYNONYMS
from exp.ours.train.runner import PredictionArg
from exp.ours.util import py_utils


class MaskSpec(PredictionArg):

  def target_eos(self):
    raise NotImplementedError()

  def get_target_words(self):
    raise ValueError()

  def get_inverse(self):
    return False


@PredictionArg.register("coco-cat-voc")
@dataclass
class CocoCategoryVoc(MaskSpec):
  val: float
  syn: bool = False
  inverse: bool = False

  def target_eos(self):
    return True

  def get_name(self):
    return "coco_category_voc"
  
  def get_target_words(self):
    if self.syn:
      raise NotImplementedError()
    return list(COCO_CATEGORIES)

  def get_inverse(self):
    return self.inverse


@dataclass
@PredictionArg.register("sce-seen-categories")
class SceSeenCategories(MaskSpec):
  task: Task
  val: float
  syn: bool=False

  def get_name(self):
    return "sce_unseen"

  def target_eos(self):
    return False

  def get_target_words(self):
    unseen = set(GpvDataset.UNSEEN_GROUPS[self.task])
    return [x for x in COCO_CATEGORIES if x in unseen]


@dataclass
@PredictionArg.register("opensce-unseen-categories")
class OpenSceUnseenCategories(MaskSpec):
  task: Task
  val: float
  syn: bool=False

  def get_name(self):
    return "sce_unseen"

  def target_eos(self):
    return False

  def get_target_words(self):
    cats = OpenSceCategories()
    all_coco_seen = [x for x in COCO_CATEGORIES if x not in set(GpvDataset.UNSEEN_GROUPS[self.task])]
    all_coco_seen = set(py_utils.flatten_list(SYNONYMS[x] for x in all_coco_seen))
    cats = [x for x in cats if x not in all_coco_seen]
    if self.syn:
      cats = py_utils.flatten_list(OPENSCE_SYNONYMS[x] for x in cats)
    return cats

@dataclass
@PredictionArg.register("sce-unseen-categories")
class SceUnseenCategories(MaskSpec):
  task: Task
  val: float
  syn: bool=False

  def get_name(self):
    return "sce_unseen"

  def target_eos(self):
    return False

  def get_target_words(self):
    words = GpvDataset.UNSEEN_GROUPS[self.task]
    if self.syn:
      words = py_utils.flatten_list(SYNONYMS[x] for x in words)
    return words


@dataclass
@PredictionArg.register("target-words")
class TargetWords(MaskSpec):
  words: List[str]
  val: float

  def get_name(self):
    return "words"

  def get_target_words(self, task):
    return self.words
