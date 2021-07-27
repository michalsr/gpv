import logging
from argparse import ArgumentParser, Action
from os.path import join
from typing import List

from exp.ours.data.dataset import GpvDataset
from exp.ours.data.gpv_data import Task, GPV1_TASKS
from exp.ours.util import our_utils
from utils.io import load_json_object


def add_dataset_args(parser: ArgumentParser, sample=True,
                     part_default=("test",), task_default=("all",)):
  parser.add_argument("--part", default=part_default,
                      choices=["val", "test", "all", "train"], nargs="+")
  parser.add_argument("--task", default=task_default,
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

  tasks = list(args.task)
  if tasks == ["all"]:
    tasks = list(Task)
  elif tasks == ["gpv1"]:
    tasks = GPV1_TASKS
  elif any(x == "train" for x in tasks):
    if train_tasks is None:
      raise ValueError()
    tasks = [Task(x) for x in tasks if x != "train"]
    tasks = list(set(tasks).union(train_tasks))
  else:
    tasks = [Task(x) for x in tasks]

  sample = None if not sample else getattr(args, "sample", None)
  to_show = []
  for task in tasks:
    for part in parts:
      to_show += [GpvDataset(task, part, split, sample=sample)]
  return to_show


# TODO this is pretty hacky
class MarkIfNotDefault(Action):

  def __init__(self, option_strings, dest, nargs=None, **kwargs):
    super().__init__(option_strings, dest, nargs, **kwargs)

  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, self.dest, values)
    assert not hasattr(namespace, f"{self.dest}_is_default")
    setattr(namespace, f"{self.dest}_not_default", True)

