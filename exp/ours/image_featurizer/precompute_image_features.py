import argparse
import logging
from collections import defaultdict
from datetime import datetime
from os import mkdir, listdir
from os.path import join, exists
from typing import Any, Union, Optional, Tuple, List

import h5py
import torch
import torchvision.ops
from dataclasses import dataclass

from torch.utils.data import DataLoader
from tqdm import tqdm

from exp.ours import file_paths
from exp.ours.data.gpv_data import Task, GPVExample
from exp.ours.experiments.datasets_cli import add_dataset_args, get_datasets_from_args
from exp.ours.image_featurizer.detr_featurizer import PretrainedDetrFeaturizer
from exp.ours.image_featurizer.image_featurizer import ImageRegionFeatures, numpy_xywh_to_cxcywh
from exp.ours.image_featurizer.vinvl_featurizer import VinvlImageFeaturizer, \
  VinVLPrecomputedFeatures, VinvlBackboneImageFeaturizer
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


def _get_model(model_name):
  if model_name == "vinvl":
    model = VinvlImageFeaturizer()
  elif model_name == "vinvl-R50C4_4setsvg":
    model = VinvlImageFeaturizer("R50C4_4setsvg")
  elif model_name == "vinvl-R50C4_4setsvg-rboxes":
    model = VinvlBackboneImageFeaturizer(
      "vinvl", "R50C4_4setsvg", None, "all")
  elif model_name == "vinvl-R50C4_4sets":
    model = VinvlImageFeaturizer("R50C4_4sets")
  elif model_name == "vinvl_precomputed":
    model = VinVLPrecomputedFeatures()
  elif model_name == "detr_coco_sce":
    model = PretrainedDetrFeaturizer(
      pretrained_model="coco_sce", clip_boxes=True, full_classifier=True)
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
    if not args.features:
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
  for _ in range(max(args.num_workers, 1)):
    for _ in devices:
      input_q.put(None)
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
  crop: Optional[Tuple[float, float, float, float]]
  query_boxes: np.ndarray

  @property
  def target_boxes(self):
    return None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("name")
  parser.add_argument("--model", default="vinvl")
  add_dataset_args(parser, part_default=("all",), task_default=("all",))
  parser.add_argument("--devices", default=None, nargs="+", type=int)
  parser.add_argument("--batch_size", default=12, type=int)
  parser.add_argument("--num_workers", default=4, type=int)
  parser.add_argument("--features", action="store_true")
  parser.add_argument("--force_dist", action="store_true")
  parser.add_argument("--output_format", choices=["directory", "hdf5", "pkl", "none"], default="hdf5")
  args = parser.parse_args()
  all_image_query = True

  py_utils.add_stdout_logger()

  output_format = args.output_format

  devices = args.devices
  if devices is None:
    devices = [our_utils.get_device()]

  logging.info("Setting up data loader")

  queries = defaultdict(set)

  # for ds in get_datasets_from_args(args, split="coco_sce"):
  #   for ex in ds.load():
  #     if ex.crop is not None:
  #       assert ex.query_box is None
  #       queries[(ex.image_id, tuple(ex.crop))].add(None)
  #     elif ex.query_box is not None:
  #       queries[(ex.image_id, None)].add(tuple(ex.query_box))
  #     else:
  #       queries[(ex.image_id, None)].add(None)

  with open(file_paths.INVALID_WEB_IMAGES_LIST) as f:
    black_list = set(f.read().split())

  for example in listdir(file_paths.WEB_DIR):
    if example not in black_list:
      queries[(example, None)].add(None)

  # print("DEBUG")
  # keys = [50518, 117601]
  # keys = [(x, None) for x in keys]
  # assert all(k in queries for k in keys)
  # queries = {k: queries[k] for k in keys}

  targets = []
  for (image_id, crop), image_queries in queries.items():
    if all_image_query:
      # An all-image box might already exist as, so add it to the set
      if image_queries == {None}:
        # Use a normalized representation so we don't have to look up the image size
        image_queries.add((0, 0, 1.0, 1.0))
      else:
        w, h = image_utils.get_image_size(image_id)
        image_queries.add((0, 0, w, h))
    image_queries = [x for x in image_queries if x is not None]
    if image_queries:
      image_queries = np.array(image_queries, dtype=np.float32)
    else:
      image_queries = None
    targets.append(ExtractionTarget(image_id, crop, image_queries))

  if output_format == "hdf5" and exists(get_hdf5_image_file(args.name)):
    logging.info("Checking for existing features...")
    # Check to see what images we have already completed
    filtered_targets = []
    with h5py.File(get_hdf5_image_file(args.name), "r") as f:
      keys = set(f.keys())
      for target in tqdm(targets, ncols=100):
        key = image_utils.get_cropped_img_key(target.image_id, target.crop)
        if key in keys:
          done = True
          # To be safe, we also check all the required fields are there
          # grp = f[key]
          # base_fields = ["bboxes", "objectness"]
          # if args.features:
          #   base_fields.append("features")
          # if any(x not in grp for x in base_fields):
          #   done = False
          # elif target.query_boxes is not None:
          #   query_fields = ["query_bboxes", "query_objectness", "query_features"]
          #   if any(x not in grp for x in query_fields):
          #     done = False
          # if not done:
          #   # TODO maybe we should just delete the offending group?
          #   raise ValueError("Incomplete image write")
        else:
          done = False

        if not done:
          filtered_targets.append(target)
    n_complete = len(targets) - len(filtered_targets)
    if len(filtered_targets) == 0:
      logging.info("All target already found in file")
      return
    if len(filtered_targets) != len(targets):
      logging.info(f"Found {n_complete}/{len(targets)} ({100*n_complete/len(targets):0.3f}%) "
                   f"complete targets, will process {len(filtered_targets)} remaining")
      targets = filtered_targets
    else:
      logging.info(f"Running on {len(targets)} images")
  else:
    logging.info(f"Running on {len(targets)} images")

  if output_format == "directory":
    out = join(file_paths.CACHE_DIR, args.name)
    logging.info(f"Saving into {out}")
    if not exists(out):
      mkdir(out)

  elif output_format == "hdf5":
    f = get_hdf5_image_file(args.name)
    logging.info(f"Saving into {f}")

    out = h5py.File(f, "a")

    # Note this just to make saving in differing formats possible
    if "box_format" not in out:
      out.create_dataset("box_format", data="xyxy")
    else:
      assert out["box_format"][()] == "xyxy"

  elif output_format == "pkl":
    f = join(file_paths.CACHE_DIR, f"{args.name}.pkl")
    logging.info(f"Saving into {f}")
    out = {}

  elif output_format == "none":
    out = None
    pass

  else:
    raise NotImplementedError(output_format)

  logging.info("Start!")

  if len(devices) == 1 and not args.force_dist:
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
      assert region_features.boxes[ix].max() <= 1.0, target.image_id
      assert region_features.boxes[ix].min() >= 0.0, target.image_id

      if target.query_boxes is not None:
        e = region_features.n_boxes[ix] - len(target.query_boxes)
      else:
        e = None if region_features.n_boxes is None else region_features.n_boxes[ix]
      to_save = dict(
        bboxes=region_features.boxes[ix, :e],
        objectness=region_features.objectness[ix, :e],
      )
      if region_features.features is not None:
        to_save["features"] = region_features.features[ix, :e]

      if target.query_boxes is not None:
        n = region_features.n_boxes[ix]
        to_save["query_bboxes"] = region_features.boxes[ix, e:n]
        to_save["query_features"] = region_features.features[ix, e:n]
        to_save["query_objectness"] = region_features.objectness[ix, e:n]

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
