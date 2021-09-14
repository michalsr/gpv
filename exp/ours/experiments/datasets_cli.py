import logging
from argparse import ArgumentParser
from os.path import join
from typing import List

from exp.ours.data.coco_segmentation import CocoSegmentationDataset
from exp.ours.data.dataset import Task, GPV1_TASKS, GPV2_TASKS
from exp.ours.data.gpv import GpvDataset
from exp.ours.data.opensce import OPENSCE_TASKS, OpenSceDataset
from exp.ours.data.webqa import WebQaDataset
from utils.io import load_json_object


def add_dataset_args(parser: ArgumentParser, sample=True,
                     part_default=("test",), task_default=("gpv2",)):
  parser.add_argument("--part", default=part_default,
                      choices=["val", "test", "all", "train"], nargs="+")
  parser.add_argument("--datasets", default=task_default,
                      required=task_default is None,
                      choices=[str(x) for x in GPV1_TASKS] +
                              ["o" + x.value for x in OPENSCE_TASKS] +
                              ["webqa-fifth", "webqa", "segmentation"] +
                              ["all", "gpv1", "gpv2", "gpv2-eval", "opensce"], nargs="+")
  if sample:
    parser.add_argument("--sample", type=int)


def get_datasets_from_args(args, model_dir=None, sample=True, trained_on_sce=None) -> List[GpvDataset]:
  if model_dir is not None and trained_on_sce is None:
    # Figure out what gpv_split the model was trained on
    trainer = load_json_object(join(model_dir, "trainer.json"))
    train_split = set()
    for ds in trainer["train_datasets"]:
      ds = ds["dataset"]
      if ds["type"] == "gpv":
        train_split.add(ds["gpv_split"])

    if len(train_split) == 0:
      trained_on_sce = None
    elif len(train_split) > 1:
      raise ValueError()
    else:
      trained_on_sce = list(train_split)[0]
  else:
    if trained_on_sce not in {"coco", "coco_sce"}:
      raise ValueError(f"Unknown split {trained_on_sce}")
    trained_on_sce = trained_on_sce == "coco_sce"

  parts = list(args.part)
  if any(x == "all" for x in parts):
    parts = ["val", "train", "test"]

  sample = None if not sample else getattr(args, "sample", None)

  to_show = []
  open_sce_tasks = set()
  gpv_tasks = set()
  segmentation = False
  webqa = False
  for dataset in args.datasets:
    if dataset == "gpv1":
      gpv_tasks.update(GPV1_TASKS)
    elif dataset == "gpv2-eval":
      if trained_on_sce is None:
        raise ValueError()
      part = "test" if trained_on_sce else "val"
      for task in GPV2_TASKS:
        to_show += [GpvDataset(task, part, trained_on_sce, sample)]
    elif dataset in {"webqa-all", "webqa"}:
      webqa = True
    elif dataset in {"seg"}:
      segmentation = True
    elif dataset == "gpv2":
      gpv_tasks.update(GPV2_TASKS)
    elif dataset == "opensce":
      open_sce_tasks.update(OPENSCE_TASKS)
    elif dataset in {x.value for x in GPV2_TASKS}:
      gpv_tasks.add(Task(dataset))
    elif dataset[0] == "o" and dataset[1:] in {x.value for x in OPENSCE_TASKS}:
      open_sce_tasks.add(Task(dataset[1:]))
    else:
      raise NotImplementedError(dataset)

  if segmentation:
    for part in parts:
      to_show += [CocoSegmentationDataset(part, sample)]
  for task in gpv_tasks:
    for part in parts:
      to_show += [GpvDataset(task, part, trained_on_sce, sample=sample)]
  for task in open_sce_tasks:
    for part in parts:
      to_show += [OpenSceDataset(task, part, sample=sample)]
  if webqa:
    for part in parts:
      to_show += [WebQaDataset(part, sample=sample)]
  return to_show

