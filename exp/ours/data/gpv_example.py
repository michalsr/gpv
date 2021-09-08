from typing import Union, List, Any, Optional, Tuple

import numpy as np

from dataclasses import dataclass

from exp.ours.data.dataset import Task


@dataclass(frozen=True)
class SegmentationLabel:
  uscrowd: int
  area: float
  segmentation: None


@dataclass(frozen=True)
class GPVExample:
  """Data representation that can be passed to GPV `collate` functions

  This representation puts the "raw" input examples for various tasks into a universal format
  so examples from different task can be jointly processed, and may encompass some pre-processing,
  like deciding what queries to use for an example, or tokenizing the text
  """

  """ID for this example that is unique among all datasets"""
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

  """Optional array boxes that are part of the query in [x, y, h, w] form"""
  segmentation_label: Optional[SegmentationLabel] = None

  """Optional key for grouping examples into batches"""
  sort_len: Optional[int] = None

  meta: Any = None

  def get_gpv_id(self):
    return self.id

