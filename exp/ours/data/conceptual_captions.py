import base64
import json
import logging
from os import listdir
from os.path import join, exists
from typing import List, Tuple, Dict, Any

import torch
import torchvision.ops
from dataclasses import dataclass

from exp.ours import file_paths
from exp.ours.data.dataset import Dataset, Task, Caption
from exp.ours.data.gpv_example import GPVExample
import numpy as np

from exp.ours.util import py_utils
from exp.ours.util.py_utils import int_to_str
from utils.io import load_json_object, dump_json_object


@dataclass
class VinVLConceptualCaption:
  """Conceptual caption with links to load VinVL pre-computed features"""

  example_id: str
  caption: str
  pred_tsv: str
  feature_tsv: str
  pred_loc: int
  feature_loc: int

  @property
  def captions(self):
    return [Caption(self.example_id, self.caption)]

  def get_gpv_id(self):
    return self.example_id

  def load_pred(self):
    with open(self.pred_tsv) as f:
      f.seek(self.pred_loc)
      pred_image_id, pred = f.readline().split("\t")
      return json.loads(pred)

  def load_features(self):
    with open(self.feature_tsv, "r+b") as f:
      f.seek(self.feature_loc)
      parts = f.readline().split(b"\t")
      n_boxes = int(parts[1])
      ex_features = np.frombuffer(
        base64.decodebytes(parts[2]),
        dtype=np.float32).reshape((n_boxes, -1))
      return ex_features.copy()

  def load_all(self):
    features = self.load_features()
    box_sizes = features[:, 2048:2052]
    features = torch.as_tensor(features[:, :2048])

    boxes = torchvision.ops.box_convert(torch.as_tensor(box_sizes), "xyxy", "cxcywh")
    pred = self.load_pred()
    obj = torch.log(torch.as_tensor([x["conf"] for x in pred["objects"]], dtype=torch.float32))
    return features, boxes, obj


@Dataset.register("conceptual-captions")
class ConceptualCaptionsDataset(Dataset):

  def __init__(self, sample=None):
    self.sample = sample

  def get_task(self) -> Task:
    return Task.CAPTIONING

  def get_name(self) -> str:
    name = "cc"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List[VinVLConceptualCaption]:
    examples = []

    logging.info("Loading cc annotations")
    anno_src = join(file_paths.CONCEPTUAL_CAPS, "0", "annotations")
    raw_captions_file = join(anno_src, "raw_captions.json")
    if exists(raw_captions_file):
      captions = load_json_object(raw_captions_file)
    else:
      logging.info("Caching raw captions...")
      all_dataset = join(anno_src, "dataset_cc.json")
      all_dataset = load_json_object(all_dataset)
      captions = {}
      for val in all_dataset["images"]:
        if len(val["sentences"]) > 1:
          print(val)
        else:
          captions[val["imgid"]] = val["sentences"][0]["raw"]
      del all_dataset
      dump_json_object(captions, raw_captions_file)

    for subdir in sorted(listdir(file_paths.CONCEPTUAL_CAPS)):
      src = join(file_paths.CONCEPTUAL_CAPS, subdir)
      logging.info(f"Loading conceptual-caps {src}")
      feature_file = join(src, "features.tsv")
      pred_file = join(src, "predictions.tsv")
      feature_lineidx = []
      with open(join(src, "features.lineidx")) as f:
        for line in f:
          feature_lineidx.append(int(line))

      pred_lineidx = []
      with open(join(src, "predictions.lineidx")) as f:
        for line in f:
          pred_lineidx.append(int(line))

      with open(join(src, "imageid2idx.json")) as f:
        image_id_to_idx = json.load(f)

      for image_id, idx in image_id_to_idx.items():
        image_id = int(image_id)
        examples.append(VinVLConceptualCaption(
          f"cc-{image_id}", captions[str(image_id)], pred_file, feature_file,
          pred_lineidx[idx], feature_lineidx[idx]
        ))
        if self.sample is not None and len(examples) == self.sample:
          return examples

    return examples


def main():
  py_utils.add_stdout_logger()
  print("Start...")
  data = ConceptualCaptionsDataset(sample=100).load()
  print(len(data[0].load_pred()["objects"]))
  print(data[0].load_features())
  print(f"Done {len(data)}")


if __name__ == '__main__':
  main()

