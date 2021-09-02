import logging
from argparse import ArgumentParser
from os.path import join
from typing import List

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
                      choices=[str(x) for x in Task] +
                              ["o" + x.value for x in OPENSCE_TASKS] +
                              ["webqa-fifth"] +
                              ["all", "gpv1", "gpv2", "gpv2-eval", "opensce"], nargs="+")
  if sample:
    parser.add_argument("--sample", type=int)


def get_datasets_from_args(args, model_dir=None, sample=True, split=None) -> List[GpvDataset]:
  if model_dir is not None and split is None:
    # Figure out what gpv_split the model was trained on
    trainer = load_json_object(join(model_dir, "trainer.json"))
    train_split = set()
    for ds in trainer["train_datasets"]:
      ds = ds["dataset"]
      if ds["type"] == "gpv":
        train_split.add(ds["gpv_split"])

    if len(train_split) != 1:
      raise ValueError()
    split = list(train_split)[0]
  else:
    if split not in {"coco", "coco_sce"}:
      raise ValueError(f"Unknown split {split}")
    split = split == "coco_sce"

  parts = list(args.part)
  if any(x == "all" for x in parts):
    parts = ["val", "train", "test"]

  sample = None if not sample else getattr(args, "sample", None)

  to_show = []
  open_sce_tasks = set()
  gpv_tasks = set()
  webqa_names = set()
  for dataset in args.datasets:
    if dataset == "gpv1":
      gpv_tasks.update(GPV1_TASKS)
    elif dataset == "gpv2-eval":
      part = "test" if split else "val"
      for task in GPV2_TASKS:
        to_show += [GpvDataset(task, part, split, sample)]
    elif dataset == "webqa-fifth":
      webqa_names.add("fifth")
    elif dataset in {"webqa-v1"}:
      webqa_names.add("all-v1")
    elif dataset in {"webqa-all", "webqa"}:
      webqa_names.add("all-v2")
    elif dataset == "webqa-80":
      webqa_names.add("80")
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


  for task in gpv_tasks:
    for part in parts:
      to_show += [GpvDataset(task, part, split, sample=sample)]
  for task in open_sce_tasks:
    for part in parts:
      to_show += [OpenSceDataset(task, part, sample=sample)]
  for name in webqa_names:
    for part in parts:
      to_show += [WebQaDataset(name, part, sample=sample)]
  return to_show

