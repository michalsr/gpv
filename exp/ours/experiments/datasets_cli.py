import logging
from argparse import ArgumentParser
from os.path import join
from typing import List

from exp.ours.data.dataset import GpvDataset
from exp.ours.data.gpv_data import Task, GPV1_TASKS, GPV2_TASKS
from utils.io import load_json_object


def add_dataset_args(parser: ArgumentParser, sample=True,
                     part_default=("test",), task_default=("all",)):
  parser.add_argument("--part", default=part_default,
                      choices=["val", "test", "all", "train"], nargs="+")
  parser.add_argument("--datasets", default=task_default,
                      required=task_default is None,
                      choices=[str(x) for x in Task] + ["all", "gpv1", "gpv2"], nargs="+")
  if sample:
    parser.add_argument("--sample", type=int)


def get_datasets_from_args(args, model_dir=None, sample=True, split=None) -> List[GpvDataset]:
  if model_dir is not None and split is None:
    trainer = load_json_object(join(model_dir, "trainer.json"))
    train_split = set()
    train_tasks = set()
    for ds in trainer["train_datasets"]:
      ds = ds["dataset"]
      assert ds["type"] == "gpv"
      train_split.add(ds["gpv_split"])
      train_tasks.add(Task(ds["task"]))

    if len(train_split) != 1:
      raise ValueError()
    split = list(train_split)[0]
  else:
    train_tasks = None
    if split not in {"coco", "coco_sce"}:
      raise ValueError(f"Unknown split {split}")
    split = split == "coco_sce"

  parts = list(args.part)
  if any(x == "all" for x in parts):
    parts = ["val", "train", "test"]

  gpv_tasks = set()
  for dataset in args.datasets:
    if dataset == "gpv1":
      gpv_tasks.update(GPV1_TASKS)
    elif dataset == "gpv2":
      gpv_tasks.update(GPV2_TASKS)
    elif dataset in {x.value for x in GPV2_TASKS}:
      gpv_tasks.add(Task(dataset))
    else:
      raise NotImplementedError(dataset)

  sample = None if not sample else getattr(args, "sample", None)
  to_show = []
  for task in sorted(gpv_tasks):
    for part in parts:
      to_show += [GpvDataset(task, part, split, sample=sample)]
  return to_show

