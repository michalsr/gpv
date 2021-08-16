import json
from os.path import join
from typing import List

import torch
import torchvision

from exp.ours import file_paths
from exp.ours.data.dataset import Dataset, Task, VqaExample, ClsExample, CaptioningExample, Caption, \
  LocalizationExample
from exp.ours.models.model import PredictionArg
from utils.io import load_json_object
import numpy as np


@PredictionArg.register("webqa-list")
class WebQa80Answers(PredictionArg, list):
  def __init__(self):
    with open(file_paths.WEBQA80_ANSWERS) as f:
      super().__init__(json.load(f))


OPENSCE_SYNONYMS = {k: set(v) for k, v in load_json_object(file_paths.OPENSCE_SYN).items()}


def convert_xyxy_to_xywh(box):
  x1, y1, x2, y2 = box
  return x1, y1, x2-x1, y2-y1


@PredictionArg.register("opensce-categories")
class OpenSceCategories(PredictionArg, list):
  def __init__(self, synonyms=False):
    categories = set()
    with open(file_paths.OPENSCE_CATS) as f:
      for line in f:
        categories.update(x.lower() for x in line.strip().split(","))
    self.synonyms = synonyms
    if self.synonyms:
      raise NotImplementedError()
    else:
      super().__init__(list(categories))


@Dataset.register("opensce")
class OpenSceDataset(Dataset):

  FILE_PATHS = {
    Task.CAPTIONING: "captioning_samples.json",
    Task.CLS: "categorization_samples.json",
    Task.CLS_IN_CONTEXT: "categorization_samples.json",
    Task.DETECTION: "localization_samples.json",
    Task.VQA: "vqa_samples.json"
  }

  def __init__(self, task: Task, part: str, sample: int=None):
    if part not in {"test", "val"}:
      raise ValueError(part)
    self.task = task
    self.part = part
    self.sample = sample

  def get_task(self) -> Task:
    return self.task

  def get_answer_options(self, synonyms=False):
    if self.task not in {Task.CLS, Task.CLS_IN_CONTEXT}:
      raise ValueError()
    return OpenSceCategories()

  def get_name(self) -> str:
    return f"opensce-{self.part}-{self.task.value}"

  def load(self) -> List:
    src = join(file_paths.OPENSCE_SAMPLES, self.part, self.FILE_PATHS[self.task])
    data = load_json_object(src)

    out = []
    for i, ex in enumerate(data):
      ex_input = ex["input"]
      image_id = ex_input["image_id"]
      if self.task == Task.VQA:
        out.append(VqaExample(
          f"opensce-vqa-{i}", f"{self.part}/visual_genome/{image_id}.jpg", ex_input["prompt"], ex["output"]["text"],
          meta={"gpv1-unseen": ex["meta"]["category"]}))
      elif self.task in {Task.CLS, Task.CLS_IN_CONTEXT}:
        query_box = ex_input["task_coordinates"]
        assert len(query_box) == 1

        if self.task == Task.CLS_IN_CONTEXT:
          query_box = convert_xyxy_to_xywh(query_box[0])
          crop = None
        else:
          crop = convert_xyxy_to_xywh(query_box[0])
          query_box = None
        out.append(ClsExample(
          f"opensce-cls-{i}", self.task, f"{self.part}/open_images/{image_id}.jpg",
          ex["output"]["text"], query_box=query_box, crop=crop,
          meta={"gpv1-unseen": ex["meta"]["category"], "gpv1-prompt": ex_input["prompt"]}))
      elif self.task == Task.CAPTIONING:
        captions = Caption(f"opensce-cap-{i}", None, meta=None)
        out.append(CaptioningExample(f"opensce-cap-{i}", f"{self.part}/nocaps/{image_id}.jpg", [captions], meta=None))
      elif self.task == Task.DETECTION:
        outputs = np.array(ex["output"]["coordinates"])

        # Convert from xyxy to expected xywy
        boxes = torchvision.ops.box_convert(torch.as_tensor(outputs), "xyxy", "xywh").numpy()

        out.append(LocalizationExample(
          f"opensce-loc-{i}", f"{self.part}/open_images/{image_id}.jpg", boxes,
          ex_input["task_text"], meta={"gpv1-prompt": ex_input["prompt"]}
        ))
      else:
        raise RuntimeError(self.task)

    if self.sample is not None:
      out.sort(key=lambda x: x.gpv_id)
      np.random.RandomState(626).shuffle(out)
      out = out[:self.sample]

    return out


OPENSCE_TASKS = list(OpenSceDataset.FILE_PATHS.keys())
