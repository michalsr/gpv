from os.path import exists

import torch
from omegaconf import OmegaConf

from exp.gpv.models.losses import GPVCriterion
from exp.ours.data.dataset import GpvDataset, Task
from exp.ours.models.gpv1 import GPV1, FULLNAMES
from exp.ours.models.losses import GPV1Loss
from exp.ours.util import our_utils
import numpy as np

from utils.io import load_pickle_object, dump_pickle_object


def get_loss_input(tasks):
  task_str = "-".join(str(x) for x in tasks)
  cache_file = f"/tmp/{task_str}-loss-out2.pkl"
  if exists(cache_file):
    return load_pickle_object(cache_file, compress=False)

  device = our_utils.get_device()

  print("Loading model...")
  conf = OmegaConf.load("configs/exp/gpv.yaml").model
  model = GPV1(conf)
  model.initialize()
  model.to(device)

  rng = np.random.RandomState(1312)

  data = []
  for task in tasks:
    print(f"Loading data for {task}...")
    examples = GpvDataset(task, "val", True).load()
    rng.shuffle(examples)
    data += examples[:2]

  print(f"Running...")
  prep = [model.preprocess_example(x) for x in data]
  inputs = model.get_collate_train()(prep)
  targets = inputs.pop("targets")
  with torch.no_grad():
    out = model(**inputs)

  dump_pickle_object((out, targets), cache_file, compress=False)
  return out, targets


FULLNAMES_MAP = {v: k for k, v in FULLNAMES.items()}


def main():
  tasks = [Task.DETECTION]
  outputs, targets = get_loss_input(tasks)

  conf = OmegaConf.load("configs/exp/gpv.yaml").losses
  criterion = GPVCriterion(conf)
  total_loss, loss_dict = criterion(outputs, targets)
  print(total_loss, loss_dict)

  # for k, v in outputs.items():
  #   print(k, v.size())
  # outputs["detr_hs"] = outputs["detr_hs"].squeeze(0)
  # print(targets)
  tasks = [FULLNAMES_MAP[x["task"]] if "task" in x else Task.DETECTION for x in targets]
  labels = torch.stack([x["answer_token_ids"] for x in targets], 0)
  ours = GPV1Loss()
  our_loss, our_dict = ours(
    outputs["answer_logits"].squeeze(0), labels,
    outputs["pred_boxes"], outputs["pred_relevance_logits"], targets, tasks
  )
  print(our_loss, our_dict)

if __name__ == '__main__':
  main()