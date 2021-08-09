import json
from typing import List

from allennlp.common import Registrable
from dataclasses import dataclass

from data.coco.synonyms import SYNONYMS
from exp.ours import file_paths
from exp.ours.data.dataset import GpvDataset
from exp.ours.data.gpv_data import Task
from exp.ours.data.source_data import ID_TO_COCO_CATEGORY
from exp.ours.train.runner import PredictionArg
from exp.ours.util import py_utils


class MaskSpec(PredictionArg):

  def target_eos(self):
    raise NotImplementedError()

  def get_target_words(self):
    raise ValueError()

  def get_inverse(self):
    return False


@PredictionArg.register("coco-categories")
class CocoCategories(PredictionArg, list):
  def __init__(self):
    super().__init__(ID_TO_COCO_CATEGORY.values())


@PredictionArg.register("webqa-list")
class WebQa80Answers(PredictionArg, list):
  def __init__(self):
    with open(file_paths.WEBQA80_ANSWERS) as f:
      super().__init__(json.load(f))


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
    return list(ID_TO_COCO_CATEGORY.values())

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
    return [x for x in ID_TO_COCO_CATEGORY.values() if x in unseen]


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
