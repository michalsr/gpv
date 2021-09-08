import logging
from os.path import join
from typing import Union, Optional, Dict, List, Any

import numpy as np
from dataclasses import dataclass

from exp.ours import file_paths
from exp.ours.data.dataset import Dataset, Task
from exp.ours.data.gpv import COCO_ID_TO_CATEGORY
from exp.ours.util import py_utils
from exp.ours.util.py_utils import int_to_str
from utils.io import load_json_object


@dataclass(frozen=True)
class SegmentationExample:
  gpv_id: str
  image_id: Union[int, str]
  category: str
  area: float
  segmentation: Any
  iscrowd: int
  meta: Optional[Dict] = None

  def get_gpv_id(self):
    return self.gpv_id


@Dataset.register("coco-segmentation")
class CocoSegmentationDataset(Dataset):
  """Loads coco segmentation data in its on-disk format"""

  def __init__(self, split: str, sample=None):
    if split not in {"train", "val"}:
      raise ValueError()
    self.split = split
    self.sample = sample

  def get_name(self) -> str:
    name = f"coco2015-seg-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_task(self) -> Task:
    return Task.SEGMENTATION

  def load(self) -> List[SegmentationExample]:
    src = join(file_paths.COCO_ANNOTATIONS, f"instances_{self.split}2014.json")
    logging.info(f"Loading segmentation examples from {src}")
    data = load_json_object(src)["annotations"]
    out = []
    for anno in data:
      out.append(SegmentationExample(
        f"coco-seg{anno['id']}",
        anno["image_id"],
        COCO_ID_TO_CATEGORY[anno['category_id']],
        anno["area"],
        anno["segmentation"],
        anno["iscrowd"],
      ))
    return out


if __name__ == '__main__':
  py_utils.add_stdout_logger()
  CocoSegmentationDataset("val").load()



