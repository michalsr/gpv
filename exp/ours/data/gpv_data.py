import enum
import torch
from typing import Union, Dict, List, Any, Optional, Tuple

import numpy as np
from allennlp.common import FromParams

from dataclasses import dataclass


class Task(FromParams, enum.Enum):
  CLS = "cls"
  VQA = "vqa"
  DETECTION = "det"
  CAPTIONING = "cap"
  CLS_IN_CONTEXT = "cic"
  WEBQA = "webqa"

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

  def to(self, device):
    return self


GPV1_TASKS = [Task.CLS, Task.VQA, Task.CAPTIONING, Task.DETECTION]

GPV2_TASKS = [Task.CLS_IN_CONTEXT, Task.CLS, Task.VQA, Task.CAPTIONING, Task.DETECTION]


@dataclass(frozen=True)
class GPVExample:
  """Data representation that can be passed to GPV `collate` functions

  This representation puts the "raw" input examples for various tasks into a universal format
  so examples from different task can be jointly processed, and may encomposs some pre-processing,
  like deciding what queries to use for an example, or tokenizing the text
  """

  id: str

  """Task this example if for"""
  task: Task

  """Image this is for"""
  image_id: Union[str, int]

  """Query (or list of queries) that can be used for this example, possibly tokenized"""
  query: Union[str, List[str], List[np.ndarray]]

  """Boxes to predict for this query, if there are any, in [x, y, h, w] form"""
  target_boxes: Optional[np.ndarray] = None

  """Text to produce for this example, if there is any"""
  target_answer: Optional[Any] = None

  """Optionally for a crop of the image in [x, y, h, w] form"""
  crop: Optional[Tuple[float, float, float, float]] = None

  """Optional array boxes that are part of the query in [x, y, h, w] form"""
  query_boxes: np.ndarray = None

  """Optional key for grouping examples into batches"""
  sort_len: Optional[int] = None

  meta: Any = None

  def get_gpv_id(self):
    return self.id
