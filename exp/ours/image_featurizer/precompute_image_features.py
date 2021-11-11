import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime
from os import mkdir, listdir
from os.path import join, exists, relpath
from typing import Any, Union, Optional, Tuple, List

import h5py
import torch
import torchvision.ops
from dataclasses import dataclass

from torch.utils.data import DataLoader
from tqdm import tqdm

from exp.ours import file_paths
from exp.ours.data.gpv import GpvDataset
from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.dataset import Task, ClsExample, GPV2_TASKS
from exp.ours.data.opensce import OPENSCE_TASKS, OpenSceDataset
from exp.ours.data.webqa import WebQaDataset
from exp.ours.experiments.datasets_cli import add_dataset_args, get_datasets_from_args
from exp.ours.image_featurizer.detectron_detectors import DetectronObjectDetector
from exp.ours.image_featurizer.detr_featurizer import PretrainedDetrFeaturizer
from exp.ours.image_featurizer.fatser_rcnn import FasterRCNNFeaturizer
from exp.ours.image_featurizer.image_featurizer import ImageRegionFeatures
from exp.ours.image_featurizer.vinvl_featurizer import VinvlImageFeaturizer, \
  VinVLPrecomputedFeatures
from exp.ours.util import our_utils, py_utils, image_utils
import numpy as np

from exp.ours.util.image_utils import get_hdf5_image_file
from exp.ours.util.our_utils import QueueDataset
from utils.io import dump_pickle_object


@dataclass
class WrapCollate:
  collater: Any

  def __call__(self, batch: 'List[ExtractionTarget]'):
    try:
      features = self.collater.collate(batch)[0]
    except Exception as e:
      # Mulit-processing is sometimes flakey about show us the right error
      # so print here just to ensure it is visible
      print(f"Error collating examples {[x.image_id for x in batch]}")
      print(e)
      logging.error(f"Error collating examples {[x.image_id for x in batch]}")
      raise e
    return batch, features


DETECTRON_MODELS = {
  "R50-C4-1x": "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
  "R50-FPN-1x": "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
  "R50-DC5-1x": "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
  "R50-FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
  "R50-C4": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
  "R101-C4": "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
  "R101-DC5": "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
  "R101-FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
  "X101-FPN": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
}

def _get_model(model_name):
  if model_name == "vinvl":
    model = VinvlImageFeaturizer()
  elif model_name == "vinvl-R50C4_4setsvg":
    model = VinvlImageFeaturizer("R50C4_4setsvg")
  elif model_name == "vinvl-R50C4_4sets":
    model = VinvlImageFeaturizer("R50C4_4sets")
  elif model_name == "vinvl_precomputed":
    model = VinVLPrecomputedFeatures()
  elif model_name == "detr_coco_sce":
    model = PretrainedDetrFeaturizer(
      pretrained_model="coco_sce", clip_boxes=True, full_classifier=True)
  elif model_name == "detr_coco":
    model = PretrainedDetrFeaturizer(
      pretrained_model="coco", clip_boxes=True, full_classifier=True)
  # elif model_name == "faster_rcnn":
  #   model = FasterRCNNFeaturizer()
  elif model_name in DETECTRON_MODELS:
    model = DetectronObjectDetector(DETECTRON_MODELS[model_name])
    model.model.roi_heads.box_predictor.test_score_thresh = 1e-5

  elif model_name.startswith("1class-detectron-rcnn"):
    model = DetectronObjectDetector(DETECTRON_MODELS[model_name], pretrained=False)
    model.model.roi_heads.box_predictor.test_score_thresh = 1e-5
    checkpoint_num = int(model_name.split("--")[-1])
    state = join(file_paths.one_class_r50_c4, f"model_00{checkpoint_num}.pth")
    logging.info(f"Loading state from {state}")
    state = torch.load(state)["model"]
    model.model.load_state_dict(state)

  elif model_name == "dbg":
    model = DetectronObjectDetector(DETECTRON_MODELS["R50-C4"], pretrained=False, one_class=True)
    model.model.roi_heads.box_predictor.test_score_thresh = 1e-5
    state = join("/var/chrisc/detectron2/models/fc50-c4-1x-1class", "model_final.pth")
    logging.info(f"Loading state from {state}")
    print(f"Loading state from {state}")
    state = torch.load(state)["model"]
    model.model.load_state_dict(state)

  elif model_name == "faster_rcnn_1class":
    model = FasterRCNNFeaturizer("fasterrcnn_resnet50_fpn", "models/detectors/dector-rcnn-base-1class/model_25bg.pth")
    model.model.roi_heads.box_predictor.box_score_thresh = 1e-5
  elif model_name == "faster_rcnn2":
    model = FasterRCNNFeaturizer(
      "resnext101_32x8d",
      "model/dector-complete/resnext101-toplayer/model_25.pth")
  else:
    raise NotImplementedError(model_name)
  model.eval()
  return model


def _run(device, data, args):
  model = _get_model(args.model)
  model.to(device)

  loader = DataLoader(
    data, batch_size=args.batch_size, shuffle=False,
    collate_fn=WrapCollate(model.get_collate(False)), num_workers=args.num_workers)
  for examples, batch in loader:
    batch = our_utils.to_device(batch, device)
    with torch.no_grad():
      out: ImageRegionFeatures = model(**batch)
    if args.no_features:
      out.features = None
    batch, n = out.boxes.size()[:2]
    boxes = torchvision.ops.box_convert(out.boxes.view(-1, 4), "cxcywh", "xyxy")
    out.boxes = boxes.view(batch, n, 4)
    yield examples, out.numpy()


def _run_worker(rank, devices, input_q, out_q, args):
  dataset = QueueDataset(input_q)
  for x in _run(devices[rank], dataset, args):
    out_q.put(x)


def _run_dist(targets, devices, args):
  # We need to use a spawn context to be compatible with `torch.multiprocessing.spawn`
  ctx = torch.multiprocessing.get_context("spawn")
  input_q = ctx.Queue()
  for x in targets:
    input_q.put(x)
  out_q = ctx.Queue()

  args = (devices, input_q, out_q, args)
  context = torch.multiprocessing.spawn(_run_worker, nprocs=len(devices), args=args, join=False)

  seen = 0
  while True:
    k, v = out_q.get()
    seen += len(k)

    if seen == len(targets):
      assert input_q.empty()
      yield k, v
      break
    else:
      yield k, v

  while not context.join():
    pass

  return


@dataclass(frozen=True)
class ExtractionTarget:
  """Target to extract features for, duck-types GPVExample"""
  image_id: Union[str, int]
  image_file: Optional[str]=None
  crop: Optional[Tuple[float, float, float, float]]=None
  query_boxes: Optional[np.ndarray]=None

  @property
  def target_boxes(self):
    return None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("image_source", help="Directory containing images or name of dataset")
  parser.add_argument("output", help="file to save the output")
  parser.add_argument("--dataset_name", help="Name for this set of images")
  parser.add_argument("--no_query", action="store_true")
  parser.add_argument("--no_features", action="store_true")
  parser.add_argument("--append", action="store_true")
  parser.add_argument("--model", default="vinvl")
  parser.add_argument("--devices", default=None, nargs="+", type=int)
  parser.add_argument("--batch_size", default=12, type=int)
  parser.add_argument("--num_workers", default=4, type=int)
  parser.add_argument("--output_format", choices=["directory", "hdf5", "pkl", "none"], default="hdf5")
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  output_format = args.output_format

  devices = args.devices
  if devices is None:
    devices = [our_utils.get_device()]

  logging.info(f"Gathering images from {args.image_source}")

  default_query_box = None if args.no_query else (0, 0, 1.0, 1.0)  # Always get features for the all-image box
  targets: List[ExtractionTarget] = []

  if args.image_source in {"hico", "hico-dbg"}:
    from exp.ours.data.hico import HicoDataset
    images = []
    for split in (["val", "train", "test"] if args.image_source == "hico" else ["val"]):
      for ex in HicoDataset(split).load():
        qboxes = (
            py_utils.flatten_list(h.object_bboxes for h in ex.hois) +
            py_utils.flatten_list(h.human_bboxes for h in ex.hois) +
            [[0, 0, ex.image_size[1], ex.image_size[0]]]
        )
        query_boxes = np.stack(qboxes, 0)
        query_boxes = torchvision.ops.box_convert(torch.as_tensor(query_boxes), "xyxy", "xywh")
        images.append((ex.image_id, query_boxes))
    assert len(set(x[0] for x in images)) == len(images)
    for image_id, qboxes in images:
      targets.append(ExtractionTarget(image_id, query_boxes=qboxes.float().numpy()))

  elif args.image_source == "web":
    queries = defaultdict(set)
    for split in ["train", "test", "val"]:
      for ex in WebQaDataset(split).load():
        queries[(ex.image_id, None)].add(default_query_box)
    for (image_id, crop), parts in queries.items():
      parts = [x for x in parts if x is not None]
      qboxes = np.array(parts, dtype=np.float32) if parts else None
      targets.append(ExtractionTarget(image_id, None, crop, qboxes))
    logging.info(f"Running on {len(targets)} images")

  elif args.image_source.startswith("opensce"):
    logging.info("Running on OpenSCE")
    if args.image_source == "opensce":
      tasks = GPV2_TASKS
    else:
      tasks = [Task(args.image_source.split("-")[-1])]
    queries = defaultdict(set)
    for task in tasks:
      for part in ["val", "test"]:
        for ex in OpenSceDataset(task, part).load():
          crop, qbox = None, default_query_box
          if task == Task.CLS_IN_CONTEXT:
            qbox = tuple(ex.query_box)
          elif task == Task.CLS:
            crop = tuple(ex.crop)
          queries[(ex.image_id, crop)].add(qbox)

    for (image_id, crop), parts in queries.items():
      parts = [x for x in parts if x is not None]
      qboxes = np.array(parts, dtype=np.float32) if parts else None
      targets.append(ExtractionTarget(image_id, None, crop, qboxes))
    logging.info(f"Running on {len(targets)} images")

  elif args.image_source == "coco-sce-nocic":
    logging.info(f"Running on {args.image_source}")
    queries = defaultdict(set)
    for task in GPV2_TASKS:
      if task == Task.CLS_IN_CONTEXT:
        continue
      for part in ["train", "val", "test"]:
        for ex in GpvDataset(task, part, True).load():
          crop, qbox = None, default_query_box
          if task == Task.CLS:
            crop = tuple(ex.crop)
          queries[(ex.image_id, crop)].add(qbox)
    for (image_id, crop), parts in queries.items():
      parts = [x for x in parts if x is not None]
      qboxes = np.array(parts, dtype=np.float32) if parts else None
      targets.append(ExtractionTarget(image_id, None, crop, qboxes))
    logging.info(f"Running on {len(targets)} images")
  elif args.image_source.startswith("coco"):
    logging.info(f"Running on {args.image_source}")
    tasks = {
      "coco": GPV2_TASKS,
      "coco-vqa": [Task.VQA],
      "coco-cic": [Task.CLS_IN_CONTEXT],
    }[args.image_source]
    queries = defaultdict(set)
    for task in tasks:
      parts = ["train", "val"] if task in {Task.CLS, Task.CLS_IN_CONTEXT, Task.DETECTION} else ["train", "val", "test"]
      for part in parts:
        for ex in GpvDataset(task, part, False).load():
          crop, qbox = None, default_query_box
          if task == Task.CLS_IN_CONTEXT and not args.no_query:
            qbox = tuple(ex.query_box)
          elif task == Task.CLS:
            crop = tuple(ex.crop)
          queries[(ex.image_id, crop)].add(qbox)
    for (image_id, crop), parts in queries.items():
      parts = [x for x in parts if x is not None]
      qboxes = np.array(parts, dtype=np.float32) if parts else None
      targets.append(ExtractionTarget(image_id, None, crop, qboxes))
    logging.info(f"Running on {len(targets)} images")
  else:
    default_query_arr = np.array([default_query_box])
    assert default_query_arr.shape == (1, 4)
    for dirpath, dirnames, filenames in os.walk(args.image_source):
      for filename in filenames:
        filename = join(dirpath, filename)
        image_id = relpath(filename, args.image_source)
        assert ".." not in image_id  # Sanity check the relpath
        image_id = args.your_dataset_name + "/" + image_id
        targets.append(ExtractionTarget(image_id, filename, None, default_query_arr))

    logging.info(f"Running on {len(targets)} images")
    logging.info(f"Example image_ids:")
    for target in np.random.choice(targets, min(8, len(targets)), replace=False):
      logging.info(f"\tfile={target.image_file}, image_id={target.image_id}")

  if output_format == "directory":
    out = args.output
    logging.info(f"Saving into {out}")
    if not exists(out):
      mkdir(out)

  elif output_format == "hdf5":
    if args.append:
      out = h5py.File(args.output, "a")
      filtered = []
      for target in targets:
        key = image_utils.get_cropped_img_key(target.image_id, target.crop)
        if key not in out:
          filtered.append(target)
      logging.info(f"Already have {len(targets)-len(filtered)}/{len(targets)} images")
      targets = filtered
      if len(targets) == 0:
        return
    else:
      if exists(args.output):
        raise ValueError()
      out = h5py.File(args.output, "w")

    # Note the box format just
    if "box_format" not in out:
      out.create_dataset("box_format", data="xyxy")
    else:
      assert out["box_format"][()] == "xyxy"

  elif output_format == "pkl":
    f = args.out
    logging.info(f"Saving into {f}")
    out = {}

  elif output_format == "none":
    out = None
    pass

  else:
    raise NotImplementedError(output_format)

  logging.info("Start!")

  if len(devices) == 1:
    it = _run(devices[0], targets, args)
  else:
    it = _run_dist(targets, devices, args)

  pbar = None  # Start pbar after the first iteration so we don't interrupt it with log output
  for examples, region_features in it:
    if pbar is None:
      pbar = tqdm(total=len(targets), desc="fe", ncols=100)
    pbar.update(len(examples))
    if output_format == "none":
      continue

    for ix, target in enumerate(examples):
      if len(region_features.boxes[ix]) == 0:
        print(ix, target.image_id, region_features.boxes[ix].shape)
        for j, ex in enumerate(examples):
          print(j, region_features.boxes[ix].shape)
          print(ex)
        raise ValueError(target.image_id)
      assert region_features.boxes[ix].max() <= 1.0, target.image_id
      assert region_features.boxes[ix].min() >= 0.0, target.image_id
      if region_features.n_boxes is None:
        n_boxes = region_features.boxes[ix].shape[0]
      else:
        n_boxes = region_features.n_boxes[ix]

      if target.query_boxes is not None:
        e = n_boxes - len(target.query_boxes)
      else:
        e = n_boxes
      to_save = dict(
        bboxes=region_features.boxes[ix, :e],
        objectness=region_features.objectness[ix, :e],
      )
      if region_features.features is not None:
        to_save["features"] = region_features.features[ix, :e]

      if target.query_boxes is not None:
        to_save["query_bboxes"] = region_features.boxes[ix, e:n_boxes]
        to_save["query_features"] = region_features.features[ix, e:n_boxes]
        to_save["query_objectness"] = region_features.objectness[ix, e:n_boxes]

        # Sanity check the query boxes saved should match the input boxes
        expected_query_boxes = torchvision.ops.box_convert(torch.as_tensor(target.query_boxes), "xywh", "xyxy")
        if not torch.all(expected_query_boxes <= 1.0):  # if target.query_boxes are not normalized
          w, h = image_utils.get_image_size(target.image_id)
          expected_query_boxes /= torch.as_tensor([w, h, w, h]).unsqueeze(0)
        assert torch.abs(expected_query_boxes - to_save["query_bboxes"]).max() < 1e-6

      key = image_utils.get_cropped_img_key(target.image_id, target.crop)
      if output_format == "directory":
        image_dir = join(out, key)
        for k, data in to_save.items():
          np.save(join(image_dir, k + ".npz"), data)
      elif output_format == "hdf5":
        grp = out.create_group(key)
        for k, data in to_save.items():
          grp.create_dataset(k, data=data)
      elif output_format == "pkl":
        out[key] = to_save

  if output_format == "hdf5":
    out.close()
  elif output_format == "pkl":
    dump_pickle_object(out, f, compress_level=3)


if __name__ == '__main__':
  main()
