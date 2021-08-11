import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from os import listdir
from os.path import join, exists, isdir, dirname
from typing import Dict

from dataclasses import replace

from exp.ours.data.dataset import GpvDataset
from exp.ours.data.gpv_data import Task, ALL_TASKS
from exp.ours.data.source_data import CocoCaptions
from exp.ours.experiments.datasets_cli import add_dataset_args, get_datasets_from_args
from exp.ours.train.evaluator import vqa_score, VqaEvaluator, CaptionEvaluator, DetectionEvaluator, \
  ClsEvaluator, Evaluator, ResultKey, WebQaEvaluator
from exp.ours.train.runner import GPVExampleOutput, load_gpv_predictions
from exp.ours.util import py_utils, our_utils
from exp.ours.util.to_params import to_params
from utils.io import load_json_object, dump_json_object
import numpy as np


def find_eval_files(run_dir, prefix):
  output = defaultdict(list)
  eval_dir = join(run_dir, "eval")
  if not exists(eval_dir):
    return output
  for subdir_name in listdir(eval_dir):
    subdir = join(eval_dir, subdir_name)
    if subdir_name.startswith(prefix) and exists(join(subdir, "predictions.json")):
      eval_name = subdir_name.split("--")[-1]
      config = load_json_object(join(subdir, "config.json"))
      ds = config["dataset"]
      n_sample = (ds["sample"], ds["seen_sample"], ds["unseen_sample"])
      n_sample = sum(0 if x is None else x for x in n_sample)
      output[eval_name].append((subdir, None if n_sample == 0 else n_sample))

  def _get_order(x):
    return x[1]

  consolidated_out = {}
  for k, v in output.items():
    v.sort(key=_get_order, reverse=True)
    consolidated_out[k] = v[0][0]

  return consolidated_out


def get_eval_if_cached(eval_dir):
  cache_file = join(eval_dir, "eval.json")
  if exists(cache_file):
    cached = load_json_object(cache_file)
    if "version" in cached:
      stats_str = cached["stats"]
      stats = {}
      for k, v in stats_str.items():
        subset_name, metric_name = k.split("/")
        if subset_name == "all":
          subset_name = None
        stats[ResultKey(metric_name, subset_name)] = v
      return stats
  return None


def cache_evaluation(prefix_or_dir, evaluator, stats):
  if isdir(prefix_or_dir) and not prefix_or_dir.endswith("/"):
    prefix_or_dir += "/"
  cache_file = prefix_or_dir + "eval.json"
  to_save = {("all" if k.subset_name is None else k.subset_name) + "/" + k.metric_name: v for k, v in stats.items()}
  to_save = dict(
    stats=to_save,
    evaluator=to_params(evaluator, Evaluator),
    date=datetime.now().strftime("%m%d-%H%M%S"),
    version=1,
  )
  to_save_str = json.dumps(to_save, indent=2)
  with open(cache_file, "w") as f:
    f.write(to_save_str)


def get_evaluator(dataset):
  per_caption = True
  if isinstance(dataset, GpvDataset):
    unseen_split = dataset.split == "test" and dataset.gpv_split
  else:
    unseen_split = False

  if unseen_split:
    def get_subsets(x):
      if isinstance(x, CocoCaptions):
        if per_caption:
          target_cap = [cap for cap in x.captions if cap.id == x.image_id]
          assert len(target_cap) == 1
          is_unseen = len(target_cap[0].meta["gpv1-unseen"]) > 0
        else:
          raise NotImplementedError()
      else:
        is_unseen = len(x.meta["gpv1-unseen"]) > 0
      return ["unseen"] if is_unseen else ["seen"]
  else:
    get_subsets = None

  evaluator = {
    Task.VQA: VqaEvaluator(),
    Task.CAPTIONING: CaptionEvaluator(per_caption=per_caption, bleu=None),
    Task.DETECTION: DetectionEvaluator(),
    Task.CLS: ClsEvaluator(),
    Task.CLS_IN_CONTEXT: ClsEvaluator(),
    Task.WEBQA: WebQaEvaluator()
  }[dataset.get_task()]
  return evaluator, get_subsets


ALL_TABLE_TASK_METRICS = {
  Task.VQA: "score", Task.DETECTION: "AP", Task.CAPTIONING: "cider",
  Task.CLS: "accuracy", Task.CLS_IN_CONTEXT: "accuracy", Task.WEBQA: "accuracy",
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("prediction_file")
  add_dataset_args(parser, sample=False)

  parser.add_argument("--show_n", action="store_true")
  parser.add_argument("--eval_name", default=None)
  parser.add_argument("--sample", type=int)
  parser.add_argument("--nocache", action="store_true")
  parser.add_argument("--per_run", action="store_true")
  args = parser.parse_args()
  py_utils.add_stdout_logger()

  args.cache = not args.nocache
  if args.sample and args.cache:
    raise ValueError("Cannot sample if caching")

  # Figure out what prediction directories we are going to evaluate
  model_dirs = our_utils.find_models(args.prediction_file)

  # [model_name, eval_name, run_name, dataset_name] -> prediction dir
  target_files = {}
  for model_name, (model_dir, runs) in model_dirs.items():
    if model_name == "":
      model_name = dirname(model_dir)

    datasets = get_datasets_from_args(args, model_dir, sample=False)
    prefixes = []
    for dataset in datasets:
      prefix = dataset.get_name()
      if args.eval_name:
        prefix = prefix + "--" + args.eval_name
      prefixes.append((prefix, dataset.get_name()))

    for r_ix, run in enumerate(runs):
      for prefix, dataset_name in prefixes:
        model_files = find_eval_files(run, prefix)
        if len(model_files) == 0 and args.task != "all":
          logging.info(f"No predictions for model {model_name}/{dataset_name}: {run}")
        for k, v in model_files.items():
          if args.per_run:
            target_files[(model_name + f"/{r_ix}", k, 0, dataset_name)] = v
          else:
            target_files[(model_name, k, r_ix, dataset_name)] = v

  if len(target_files) == 0:
    print("No evals found")
    return

  # Get results that are already cached
  results = {}
  if args.cache:
    to_eval = {}
    for key, eval_dir in target_files.items():
      cached = get_eval_if_cached(eval_dir)
      if cached:
        logging.info(f"Loaded cached stats for {eval_dir}")
        results[key] = cached
      else:
        to_eval[key] = eval_dir

    if len(to_eval) == 0:
      logging.info(f"All results cached")
    elif len(results) > 0:
      logging.info(f"Had {len(results)} cached results evaluating {len(to_eval)}")
    else:
      logging.info(f"Evaluating {len(to_eval)} predictions")
  else:
    to_eval = target_files
    logging.info(f"Evaluating {len(to_eval)} predictions without caching")

  cached_datasets = {}
  name_to_dataset = {x.get_name(): x for x in datasets}

  # Evaluate results that are not cached
  for (model_name, eval_name, r_ix, ds_name), eval_dir in to_eval.items():
    logging.info(f"Evaluating {model_name}-{eval_name} at {eval_dir}")
    if ds_name not in cached_datasets:
      dataset = name_to_dataset[ds_name]
      logging.info(f"Loading data for {ds_name}")
      instances = dataset.load()
      evaluator, get_subsets = get_evaluator(dataset)
      cached_datasets[ds_name] = instances, evaluator, get_subsets
    else:
      instances, evaluator, get_subsets = cached_datasets[ds_name]

    task = dataset.get_task()
    load_boxes = task == Task.DETECTION
    pred = load_gpv_predictions(eval_dir, load_boxes)
    if args.sample:
      keys = sorted(pred.keys())
      np.random.RandomState(12312).shuffle(keys)
      pruned = {}
      for k in keys[:args.sample]:
        pruned[k] = pred[k]
      pred = pruned

    stats = evaluator.evaluate(instances, pred, subset_mapping=get_subsets,
                               allow_partial=True)
    stats[ResultKey("n", None)] = len(pred)

    if args.cache:
      cache_evaluation(eval_dir, evaluator, stats)

    results[(model_name, eval_name, r_ix, ds_name)] = {replace(k, dataset_name=ds_name): v for k, v in stats.items()}

  per_dataset = defaultdict(lambda: defaultdict(list))
  for (model_name, eval_name, r_ix, ds_name), stats in results.items():
    per_dataset[ds_name][(model_name, eval_name)].append(stats)

  all_table = defaultdict(dict)
  for ds_name, grouped in per_dataset.items():
    task = name_to_dataset[ds_name].get_task()
    if task != Task.CAPTIONING:
      factor = 100
      val_format = "%.2f"
    else:
      val_format = "%.3f"
      factor = 1
    
    for key, results in grouped.items():
      n_runs = len(results)
      results = py_utils.transpose_list_of_dicts(results)
      results = {
        str(k): np.mean(v)*(1 if k.metric_name == "n" else factor) for k, v
        in results.items() if (args.show_n or k.metric_name != "n")}
      all_table["/".join(key)].update({task.value + "/" + str(k): val_format % v for k, v in results.items()})
      if n_runs > 1:
        results["n-runs"] = n_runs
      grouped[key] = results

    to_show = []
    for (model_name, eval_name), to_show_stats in grouped.items():
      model_stats = dict(model=model_name, eval=eval_name)
      model_stats.update(to_show_stats)
      to_show.append(model_stats)

  sorted_all_table = {}
  for row_name, row in all_table.items():
    sorted_row = {}
    for task in ALL_TASKS:
      for subset in [None, "seen", "unseen"]:
        metric = ALL_TABLE_TASK_METRICS[task][task]
        k = task.value + "/" + str(ResultKey(metric, subset))
        if k in row:
          sorted_row[k] = row[k]
    sorted_all_table[row_name] = sorted_row

  print(py_utils.dict_of_dicts_as_table_str(sorted_all_table, None))




if __name__ == '__main__':
  main()
