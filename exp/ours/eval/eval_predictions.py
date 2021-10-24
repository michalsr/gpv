import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from os import listdir
from os.path import join, exists, isdir, dirname, basename
from typing import Dict, List, Iterable

import regex
from allennlp.common import Params
from dataclasses import replace

from data.coco.synonyms import SYNONYMS
from exp.ours.data.dataset import CaptioningExample, ALL_TASKS, Task, Dataset
from exp.ours.data.gpv import GpvDataset, COCO_CATEGORIES
from exp.ours.data.opensce import OpenSceDataset, OPENSCE_TASKS
from exp.ours.data.webqa import WebQaDataset
from exp.ours.experiments.datasets_cli import add_dataset_args, get_datasets_from_args
from exp.ours.train.evaluator import vqa_score, VqaEvaluator, CaptionEvaluator, \
  LocalizationEvaluator, \
  ClsEvaluator, Evaluator, ResultKey, WebQaEvaluator, OpenSceClsEvaluator, OpenSceVqaEvaluator, \
  BoostingOpenSceClsEvaluator
from exp.ours.train.runner import GPVExampleOutput, load_gpv_predictions
from exp.ours.util import py_utils, our_utils
from exp.ours.util.py_utils import FindAll
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
    if subdir_name.startswith(prefix):
      if exists(join(subdir, "predictions.json")) or exists(join(subdir, "eval.json")):
        eval_name = subdir_name.split("--")[-1]
        config = load_json_object(join(subdir, "config.json"))
        ds = config["dataset"]
        n_sample = (ds["sample"], ds.get("seen_sample", None), ds.get("unseen_sample", None))
        n_sample = sum(0 if x is None else x for x in n_sample)
        output[eval_name].append((subdir, None if n_sample == 0 else n_sample))

  if len(output) == 0 and prefix.startswith("webqa-v4-val"):
    # Keep evaluation from the older version of webqa, results are comparable
    return find_eval_files(run_dir, prefix.replace("webqa-v4-val", "webqa-v2-val"))


  def _get_order(x):
    return 1e9 if x[1] is None else x[1]

  consolidated_out = {}
  for k, v in output.items():
    v.sort(key=_get_order, reverse=True)
    consolidated_out[k] = v[0][0]

  return consolidated_out


def get_eval_if_cached(eval_dir):
  cache_file = join(eval_dir, "eval.json")
  if exists(cache_file):
    cached = load_json_object(cache_file)
    if isinstance(cached, list) and "in-domain" in cached[0]:
      # a nocaps eval file
      stats = {}
      for part in cached:
        for subset, r in part.items():
          for metric_name, val in r.items():
            stats[ResultKey(metric_name, subset)] = val
      return stats

    version = cached.get("version", 0)
    if version <= 3:
      eval_conf = load_json_object(join(eval_dir, "config.json"))
      ds = eval_conf["dataset"]
      if ds["type"] == "opensce" and ds["task"] in {"cic", "cls"} and ds["part"] == "val":
        if version <= 3:
          # Updated to include boosting scores
          return None

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
    version=4,
  )
  to_save_str = json.dumps(to_save, indent=2)
  with open(cache_file, "w") as f:
    f.write(to_save_str)


def get_evaluator(dataset):
  if isinstance(dataset, OpenSceDataset):
    if dataset.get_task() == Task.CAPTIONING:
      logging.warning("OpenSce caption eval not supported")
      return None, None
    if dataset.get_task() in {Task.CLS, Task.CLS_IN_CONTEXT}:
      unseen = GpvDataset.UNSEEN_GROUPS[Task.CLS]
      seen = set(py_utils.flatten_list(SYNONYMS[x] for x in COCO_CATEGORIES if x not in unseen))
      unseen = set(py_utils.flatten_list(SYNONYMS[x] for x in unseen))

      def get_subsets(x):
        if x.category in seen:
          return ["seen"]
        elif x.category not in unseen:
          return ["unseen"]
        else:
          return []
    else:
      get_subsets = None

    cls_eval = OpenSceClsEvaluator()
    if dataset.part == "val":
      cls_eval = BoostingOpenSceClsEvaluator(list(range(0, 20)), False, 20, cls_eval)
    return {
      Task.CLS: cls_eval,
      Task.DETECTION: LocalizationEvaluator(),
      Task.VQA: OpenSceVqaEvaluator(),
      Task.CLS_IN_CONTEXT: cls_eval
    }[dataset.get_task()], get_subsets

  if isinstance(dataset, WebQaDataset):
    def get_subsets(x):
      qtype = x.qtype
      return [qtype, qtype[1:]]
    return WebQaEvaluator(), get_subsets

  per_caption = True
  if isinstance(dataset, GpvDataset):
    unseen_split = dataset.split == "test" and dataset.gpv_split
  else:
    unseen_split = False

  if unseen_split:
    def get_subsets(x):
      if isinstance(x, CaptioningExample):
        if per_caption:
          target_cap = [cap for cap in x.captions if x.gpv_id == cap.gpv_id]
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
    Task.DETECTION: LocalizationEvaluator(),
    Task.CLS: ClsEvaluator(),
    Task.CLS_IN_CONTEXT: ClsEvaluator(),
  }[dataset.get_task()]
  return evaluator, get_subsets


ALL_TABLE_TASK_METRICS = {
  Task.VQA: ["score"],
  Task.DETECTION: ["AP"],
  Task.CAPTIONING: ["cider"],
  Task.CLS: ("accuracy", "top5-acc"),
  Task.CLS_IN_CONTEXT: ("accuracy", "top5-acc"),
}


def _build_order():
  dataset_order = []
  for part in ["val", "test"]:
    for task in [Task.CLS, Task.CLS_IN_CONTEXT, Task.DETECTION,
                 Task.VQA, Task.CAPTIONING]:
      dataset_order.append(OpenSceDataset(task, part).get_name())
  for gpv_split in [True, False]:
    for part in ["val", "test"]:
      for task in [Task.VQA, Task.CAPTIONING, Task.DETECTION,
                   Task.CLS, Task.CLS_IN_CONTEXT]:
        dataset_order.append(GpvDataset(task, part, gpv_split).get_name())
  for split in ["test", "val"]:
    dataset_order.append(WebQaDataset(split).get_name())
  return {k: i for i, k in enumerate(dataset_order)}


DATASET_ORDER = _build_order()

METRIC_ORDER = {k: i for i, k in enumerate([
  "iou",
  "top5-iou",
  "acc",
  "accuracy",
  "top5-acc",
])}


def _sort_keys(keys: Iterable[ResultKey]):
  def _order(key: ResultKey):
    return (
      DATASET_ORDER.get(key.dataset_name, 1000),
      key.dataset_name,
      "" if key.subset_name is None else key.subset_name,
      METRIC_ORDER.get(key.metric_name, 1000),
      key.metric_name
    )
  return sorted(keys, key=_order)


def sort_and_remap_sdout_keys(keys: Iterable[ResultKey]) -> Dict[ResultKey, str]:
  opensce_cap_names = {OpenSceDataset(Task.CAPTIONING, x).get_name() for x in ["val", "test"]}
  out = {}
  for key in keys:
    if key.dataset_name in opensce_cap_names:
      if key.metric_name.lower() == "cider" and key.subset_name in {"out-domain", "in-domain"}:
        out[key] = f"cider/{key.subset_name}-{key.metric_name}"
    elif key.dataset_name.startswith("opensce") and key.subset_name is not None:
      continue
    else:
      out[key] = str(key)

  return {k: out[k] for k in _sort_keys(out)}


def sort_and_remap_tsv_keys(keys: Iterable[ResultKey]) -> Dict[ResultKey, str]:
  out = {}
  keys = sorted(keys, key=lambda x: (
    x.dataset_name,
    "" if x.subset_name is None else x.subset_name,
    x.metric_name
  ))
  cur_names = set()
  for k in keys:
    name = str(k)
    name = name.replace("-val", "").replace("-test", "")
    name = name.replace("gpv-", "coco-")
    name = name.replace("gpvsce", "cocosce")
    name = name.replace("webqa-v4-basic", "webqa")
    name = name.replace("accuracy", "acc")
    if name in cur_names:
      raise ValueError(f"TSV naming error {name} for {str(k)}")
    cur_names.add(name)
    out[k] = name
  return {k: out[k] for k in _sort_keys(out)}


def get_hypers(h_name, model_dir):
  if h_name == "webqa":
    hyper = {}
    trainer_data = load_json_object(join(model_dir, "trainer.json"))
    hyper["epochs"] = str(trainer_data["epochs"])
    hyper["learning_rate"] = str(trainer_data["optimizer"]["lr"])
    hyper["batch_size"] = str(trainer_data["train_loader"]["batch_size"])

    model_data = load_json_object(join(model_dir, "model.json"))
    if model_data["type"] == "t5-gpv-per-box":
      hyper["per-box"] = "True"

    templates = model_data.get("webqa_templates", "none")

    init_from = model_data.get("initialize_from")
    hyper["webqa-pretrain"] = "none"
    if init_from is not None:
      if init_from == "models/webqa/basic-ep8/r0/best-state.pth":
        hyper["webqa-pretrain"] = "8epochs"
      else:
        raise NotImplementedError()
    if isinstance(templates, str):
      hyper["webqa_templates"] = templates
    elif templates is None:
      hyper["webqa_templates"] = "none"
    else:
      if 'oversample_questions' not in templates:
        if templates["version"] != 0:
          raise ValueError()
        hyper["webqa_templates"] = "v0"
      else:
        hyper["webqa_templates"] = f"v{templates.get('version', 1)}"

    ds0 = trainer_data["train_datasets"][0]["dataset"]
    if ds0["type"] == "gpv":
      hyper["sce_split"] = "True" if ds0["gpv_split"] else "False"

    ds = trainer_data["train_datasets"][-1]["dataset"]
    if ds["type"] == "webqa-v2":
      hyper["qtype"] = Dataset.from_params(Params(ds)).get_qtypes_name()
    elif ds["type"] == "webqa":
      hyper["qtype"] = ds.get("question_types", "none")
      builder = trainer_data.get("train_dataset_builder")
      if builder is not None:
        if builder["type"] == "partition-web-qa":
          hyper["part"] = builder["n_partitions"]
        else:
          raise NotImplementedError()
      else:
        assert ds["type"] == "webqa"
        assert ds["name"] == "all-v2"
        hyper["train-sample"] = str(trainer_data["train_datasets"][-1]["train_sample"])
    return hyper
  else:
    raise ValueError()


def val_to_str(key: ResultKey, val: float):
  if val is None:
    return "-"
  if isinstance(val, str):
    return val
  if "cap" in key.dataset_name:
    return "%.3f" % val
  else:
    return "%.3f" % (100*val)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("models", nargs="+")
  add_dataset_args(parser, sample=False)

  parser.add_argument("--show_n", action="store_true")
  parser.add_argument("--eval_name", default=None)
  parser.add_argument("--sample", type=int)
  parser.add_argument("--nocache", action="store_true")
  parser.add_argument("--per_run", action="store_true")
  parser.add_argument("--hyperparameters", choices=["none", "webqa"], default="none")
  parser.add_argument("--output_tsv")
  args = parser.parse_args()
  py_utils.add_stdout_logger()

  args.cache = not args.nocache
  if args.sample and args.cache:
    raise ValueError("Cannot sample if caching")

  # Figure out what prediction directories we are going to evaluate
  model_dirs = our_utils.find_models(args.models)

  # [model_name, eval_name, run_name, dataset_name] -> prediction dir
  target_files = {}
  name_to_dataset = {}
  for model_name, (model_dir, runs) in model_dirs.items():
    if model_name == "":
      model_name = dirname(model_dir)

    datasets = get_datasets_from_args(args, model_dir, sample=False)
    name_to_dataset.update({x.get_name(): x for x in datasets})
    prefixes = []
    for dataset in datasets:
      prefix = dataset.get_name()
      if args.eval_name:
        prefix = prefix + "--" + args.eval_name
      prefixes.append((prefix, dataset.get_name()))

    for r_ix, run in enumerate(runs):
      for prefix, dataset_name in prefixes:
        model_files = find_eval_files(run, prefix)
        if len(model_files) == 0 and prefix.startswith("webqa-v4-basic-val--"):
          # Backwards compatiblity fix
          model_files = find_eval_files(run, prefix.replace("webqa-v4-basic-val", "webqa-all-v2-val"))
          if len(model_files) > 0:
            logging.warn(f"Using old webqa results for {run}")

        if len(model_files) == 0:
          logging.info(f"No predictions for model {dataset_name}: {run}")
        for k, v in model_files.items():
          if args.per_run:
            target_files[(model_name + f"/{r_ix}", k, 0, dataset_name)] = v
          else:
            target_files[(model_name, k, r_ix, dataset_name)] = v

  if len(target_files) == 0:
    print("No evals found")
    return

  # Get results that are already cached
  # (model_name, eval_name, run_number, datset_name) -> ResultKey -> value
  results = {}
  if args.cache:
    to_eval = {}
    for key, eval_dir in target_files.items():
      cached = get_eval_if_cached(eval_dir)
      if cached:
        logging.info(f"Loaded cached stats for {eval_dir}")
        dataset_name = eval_dir.split("/")[-1].split("--")[0]
        results[key] = {replace(k, dataset_name=dataset_name): v for k, v in cached.items()}
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

  # Evaluate results that are not cached
  for (model_name, eval_name, r_ix, ds_name), eval_dir in to_eval.items():
    logging.info(f"Evaluating {model_name}-{eval_name} at {eval_dir}")
    if ds_name not in cached_datasets:
      dataset = name_to_dataset[ds_name]
      logging.info(f"Loading data for {ds_name}")
      instances = dataset.load()
      evaluator, get_subsets = get_evaluator(dataset)
      if evaluator is None:
        continue
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

  # Group results by model
  # (model_name, eval_name, dataset_name) -> run_number -> ResultKey -> value
  per_dataset = defaultdict(list)
  for (model_name, eval_name, r_ix, ds_name), stats in results.items():
    per_dataset[(model_name, eval_name, ds_name)].append(stats)

  # (model_name, eval_name) -> ResultKey -> aggregated value
  per_model_results = defaultdict(dict)
  for (model_name, eval_name, ds_name), result_list in per_dataset.items():
    n_runs = len(result_list)
    results = py_utils.transpose_list_of_dicts(result_list)
    results = {
      k: np.mean(v) for k, v
      in results.items() if (args.show_n or k.metric_name != "n")}
    if n_runs > 1:
      results[ResultKey("n-runs", dataset_name=ds_name)] = n_runs
    per_model_results[(model_name, eval_name)].update(results)

  all_keys = set(py_utils.flatten_list(result.keys() for result in per_model_results.values()))

  # model_name -> hyperparameter_key -> value
  hypers = None
  hyper_keys = None
  if args.hyperparameters == "webqa":
    hypers = {}
    model_name_to_model_dir = {k: v[0] for k, v in model_dirs.items()}
    for model_name in set(x[0] for x in per_model_results.keys()):
      model_dir = model_name_to_model_dir[model_name]
      hypers[model_name] = get_hypers(args.hyperparameters, model_dir)
    hyper_keys = set(py_utils.flatten_list(h.keys() for h in hypers.values()))
  elif args.hyperparameters != "none":
    raise ValueError(args.hyperparameters)

  if args.output_tsv:
    tsv_results = dict(per_model_results)
    for k, v in tsv_results.items():
      # Fix mis-named dataset
      for key, r in list(v.items()):
        if key.dataset_name == "webqa-all-v2-val":
          del v[key]
          v[replace(key, dataset_name="webqa-v4-basic-val")] = r
    tsv_keys = set(py_utils.flatten_list(result.keys() for result in tsv_results.values()))

    with open(args.output_tsv, "w") as f:
      key_map = sort_and_remap_tsv_keys(tsv_keys)
      header = ["name", "eval_name"] + [str(x) for x in list(key_map.values())]
      if hypers is not None:
        header += list(hyper_keys)
      f.write("\t".join(header))
      f.write("\n")
      for (model_name, eval_name), v in tsv_results.items():
        row = [model_name, eval_name] + [val_to_str(k, v.get(k, "-")) for k in key_map.keys()]
        if hypers is not None:
          row += [hypers[model_name].get(key, "-") for key in hyper_keys]
        f.write("\t".join(row))
        f.write("\n")

  remapped_keys = sort_and_remap_sdout_keys(list(all_keys))
  all_table = {}
  for (model_name, eval_name), row in per_model_results.items():
    all_table[model_name + "/" + eval_name] = {name: val_to_str(key, row[key]) for key, name in remapped_keys.items() if key in row}
  if hypers is not None:
    for row_name, v in all_table.items():
      h = hypers["/".join(row_name.split("/")[:-1])]
      h.update(v)
      all_table[row_name] = h
  print(all_table)
  print(py_utils.dict_of_dicts_as_table_str(all_table, None))


if __name__ == '__main__':
  main()
