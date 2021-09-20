import enum
from typing import List, Union, Optional, Tuple, Dict, Counter, Any

from allennlp.common import Registrable, FromParams
from dataclasses import dataclass

import numpy as np


class Task(FromParams, enum.Enum):
  CLS = "cls"
  VQA = "vqa"
  DETECTION = "det"
  CAPTIONING = "cap"
  CLS_IN_CONTEXT = "cic"
  WEBQA = "webqa"
  SEGMENTATION = "seg"

  @classmethod
  def from_params(
      cls,
      params,
      constructor_to_call=None,
      constructor_to_inspect=None,
      **extras,
  ):
    # Need a custom method here due to some interactions between
    # FromParams/Enum
    return Task(params["type"])

  def to_params(self):
    # Params objects can represent classes with no args as just strings
    # We do that here, which is easier to read and makes sure we can use
    # `Task` objects as keys in maps
    return self._value_

  def __str__(self):
    return self._value_

  def __reduce_ex__(self, protocol):
    # Adding `FromParam` makes enum.Enum think the default unpickle implementation will
    # fail, so it helpfully breaks pickling so we fail fast when saving with pickle.
    # In fact, the default unpickle implementation is fine because `FromParams` does not
    # add any state, so we do this redundant override of __reduce_ex__ so `enum.Enum` trusts
    # that the object can be pickled
    return enum.Enum.__reduce_ex__(self, protocol)

  # Support these to it can used in a dataloader
  def to(self, device):
    return self

  def pin_memory(self):
    return self


GPV1_TASKS = [Task.VQA, Task.CAPTIONING, Task.DETECTION, Task.CLS]

GPV2_TASKS = GPV1_TASKS + [Task.CLS_IN_CONTEXT]

ALL_TASKS = GPV2_TASKS + [Task.WEBQA]


class Dataset(Registrable):
  """Dataset we can train/evaluate on"""

  def get_task(self) -> Task:
    raise NotImplementedError()

  def get_answer_options(self, synonyms=False):
    raise NotImplementedError()

  def get_name(self) -> str:
    """Get the name of the dataset

    Should by uniquely identified with the set of examples `load` returns since we might
    use it for caching.
    """
    raise NotImplementedError()

  def load(self) -> List:
    """Loads the examples"""
    raise NotImplementedError()


@dataclass(frozen=True)
class ClsExample:
  gpv_id: str
  task: Task
  image_id: Union[int, str]
  category: str
  crop: Optional[Tuple[float, float, float, float]] = None
  query_box: Optional[Tuple[float, float, float, float]] = None
  meta: Optional[Dict] = None

  def __post_init__(self):
    if self.crop is not None and self.query_box is not None:
      raise ValueError("Both crop and query not supported")

  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class VqaExample:
  gpv_id: str
  image_id: Union[int, str]
  question: str
  answers: Union[str, Counter]
  meta: Optional[Dict] = None

  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class LocalizationExample:
  gpv_id: str
  image_id: Union[int, str]
  bboxes: np.ndarray
  category: str
  meta: Optional[Dict] = None

  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class Caption:
  gpv_id: str
  caption: Optional[str]
  meta: Optional[Dict[str, Any]] = None

  def get_gpv_id(self):
    return self.gpv_id


@dataclass(frozen=True)
class CaptioningExample:
  gpv_id: str
  image_id: str
  captions: List[Caption]
  meta: Optional[Dict[str, Any]] = None

  def get_gpv_id(self):
    return self.gpv_id


@dataclass
@Dataset.register("in-memory-ds")
class InMemoryDataset(Dataset):
  data: List
  task: Task
  name: str
  answer_options: Optional[Any]

  def to_params(self):
    return {}

  def get_answer_options(self, synonyms=False):
    return self.answer_options

  def get_task(self) -> Task:
    return self.task

  def get_name(self) -> str:
    return self.name

  def load(self) -> List:
    return self.data
