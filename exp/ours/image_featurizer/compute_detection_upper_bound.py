import argparse
import logging

import h5py
import torch
import torchvision
from tqdm import tqdm

from exp.ours.data.dataset import Task
from exp.ours.data.gpv import GpvDataset
from exp.ours.data.gpv_example import GPVExample
from exp.ours.image_featurizer.image_featurizer import Hdf5FeatureExtractorCollate
from exp.ours.util import image_utils, py_utils

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("features")
  parser.add_argument("--format", default="xyxy")
  parser.add_argument("--sample", default=None, type=int)
  parser.add_argument("--part", default="val")

  args = parser.parse_args()
  py_utils.add_stdout_logger()

  col = Hdf5FeatureExtractorCollate(
    image_utils.get_hdf5_image_file(args.features), args.format, return_features=False)
  instances = GpvDataset(Task.DETECTION, args.part, sample=args.sample).load()

  with h5py.File(col.source_file, "r") as f:
    filtered = [x for x in instances if image_utils.get_cropped_img_key(x.image_id, x.crop) in f]
  if len(filtered) != len(instances):
    logging.info(f"Missing {len(instances)-len(filtered)} instances, evaluating on {len(filtered)}")
    instances = filtered

  pbar = tqdm(total=len(instances), ncols=100)
  iou_upper_bound = []

  while len(instances) > 0:
    batch = instances[:50]
    instances = instances[50:]
    fe = col.collate([GPVExample("", Task.DETECTION, ex.image_id, None, ex.bboxes)
                     for ex in batch])[0]
    boxes = fe["features"].boxes
    for ex, boxes in zip(batch, boxes):
      boxes = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")
      w, h = image_utils.get_image_size(ex.image_id)
      gt = torch.as_tensor(ex.bboxes / np.array([w, h, w, h]).reshape((1, -1)))
      gt = torchvision.ops.box_convert(gt, "xywh", "xyxy")
      iou_upper_bound.append(torch.any(torchvision.ops.box_iou(gt, boxes) > 0.5, -1).float().mean().item())
      pbar.update(1)

  print(np.mean(iou_upper_bound))


if __name__ == '__main__':
  main()