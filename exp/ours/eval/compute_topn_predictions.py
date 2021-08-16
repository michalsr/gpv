
import argparse
import json
import logging
import os

import h5py

from exp.ours.boosting import SceUnseenCategories
from exp.ours.eval.eval_predictions import get_evaluator, cache_evaluation

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from os.path import join, exists, dirname
from shutil import rmtree

from allennlp.common import Registrable
from dataclasses import dataclass

from exp.ours.experiments.datasets_cli import add_dataset_args, get_datasets_from_args
from exp.ours.train.evaluator import ResultKey
from exp.ours.util import our_utils, py_utils, image_utils
from exp.ours.data.dataset import Dataset, Task
from exp.ours.train.runner import BeamSearchSpec, save_gpv_output, \
  run, prediction_args_to_json
from exp.ours.util.to_params import to_params


@dataclass
class EvaluationConfig(Registrable):
  beam_size: int
  max_seq_len: int
  unseen_concept_boost: float
  seen_concept_sub: float


# These make sense for T5 based on the train data, maybe not for other models
DEFAULT_MAX_SEQ_LEN = {
  Task.VQA: 20,
  Task.CLS: 8,
  Task.CLS_IN_CONTEXT: 8,
  Task.WEBQA: 8,
  Task.CAPTIONING: 30
}


def eval_on(args, run_dir, dataset, devices, skip_existing=True):
  if args.output_dir:
    output_dir = args.output_dir

  elif args.output_name:
    name = f"{dataset.get_name()}--{args.output_name}"
    eval_dir = join(run_dir, "eval")
    if not exists(eval_dir):
      os.mkdir(eval_dir)
    output_dir = join(eval_dir, name)
  else:
    output_dir = None

  if output_dir is not None:
    if exists(output_dir):
      if len(os.listdir(output_dir)) > 0:
        if skip_existing:
          logging.info(f"{output_dir} already exists, skipping")
          return

        if args.override or py_utils.get_yes_no(f"{output_dir} exists, delete (y/n)?"):
          logging.info(f"Deleting {output_dir}")
          rmtree(output_dir)
        else:
          logging.info("No override, not stopping")
          return
    elif not exists(dirname(output_dir)):
      raise ValueError(f"Parent folder {dirname(output_dir)} does not exist")
    else:
      logging.info(f"Will save to {output_dir}")
  else:
    logging.info(f"Not saving the output")

  if output_dir:
    if not exists(output_dir):
      os.mkdir(output_dir)
    logging.info(f"Saving output to {output_dir}")

  task = dataset.get_task()

  logging.info("Setting up...")
  examples = dataset.load()

  if args.max_seq_len:
    max_seq_len = args.max_seq_len
  elif task == Task.DETECTION:
    max_seq_len = None
  else:
    max_seq_len = DEFAULT_MAX_SEQ_LEN[task]
    logging.info(f"Defaulting to max_seq_len {max_seq_len} for task {task}")

  if max_seq_len is not None:
    bs = BeamSearchSpec(beam_size=args.beam_size, max_seq_len=max_seq_len)
  else:
    bs = None
  prediction_args = dict(beam_search_spec=bs)

  if args.boost_unseen:
    prediction_args["mask"] = SceUnseenCategories(task, args.boost_unseen, args.boost_syn)

  if task in {Task.CLS, Task.CLS_IN_CONTEXT, Task.WEBQA} and args.cls_mask != "none":
    logging.info("Using classification mask")
    prediction_args["answer_options"] = dataset.get_answer_options(args.cls_mask == "synonyms")

  if args.dry_run:
    logging.info("Skipping running the model since this is a dry run")
    return

  output = run(
    run_dir, examples, devices, args.batch_size, args.num_workers,
    prediction_args, beams_to_keep=args.beams_to_keep)

  if output_dir is not None:
    logging.info(f"Saving output to {output_dir}")
    save_gpv_output(output, output_dir)

    config = dict(
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      predictions_args=prediction_args_to_json(prediction_args),
      dataset=to_params(dataset, Dataset),
      date=datetime.now().strftime("%m%d-%H%M%S"),
    )

    with open(output_dir + "/config.json", "w") as f:
      json.dump(config, f, indent=2)

  if args.eval:
    logging.info("Evaluating...")
    evaluator, subsets = get_evaluator(dataset)
    results = evaluator.evaluate(examples, output, allow_partial=True, subset_mapping=subsets)

    if output_dir is not None:
      results[ResultKey("n", None)] = len(output)
      logging.info(f"Caching evaluation to {output_dir}")
      cache_evaluation(output_dir, evaluator, results)

    if task != Task.CAPTIONING:
      factor = 100
    else:
      factor = 1
    results = {str(k): v*factor for k, v in results.items()}
    print(json.dumps(results, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  add_dataset_args(parser, task_default=("train",))
  parser.add_argument("--boost_unseen", type=float, default=None)
  parser.add_argument("--boost_syn", action="store_true")
  parser.add_argument("--cls_mask", default="categories", choices=["none", "categories", "synonyms"])
  parser.add_argument("--device", nargs="+", default=[None])
  parser.add_argument("--batch_size", type=int, default=30)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--beams_to_keep", type=int, default=1)
  parser.add_argument("--max_seq_len", type=int, default=None)
  parser.add_argument("--beam_size", type=int, default=20)
  parser.add_argument("--eval", action="store_true", help="Evaluate the results")
  parser.add_argument("--override", action="store_true")
  parser.add_argument("--output_dir")
  parser.add_argument("--output_name")
  parser.add_argument("--dry_run", action="store_true")
  parser.add_argument("--nms", type=float, default=None)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.output_dir and args.output_name:
    raise ValueError("Cannot specify output_name and output_dir")

  models = our_utils.find_models(args.model)
  if len(models) == 0:
    logging.info("No models selected")
    return

  devices = our_utils.get_devices(args.device)
  if args.output_dir:
    models = py_utils.flatten_list(x[1] for x in models.values())
    if len(models) > 1:
      raise ValueError("Cannot use one output dir if more than one model selected!")
    model = models[0]
    datasets = get_datasets_from_args(args, model)
    if len(datasets) > 1:
      raise ValueError("Cannot use one output dir if more than one dataset is selected!")
    if len(datasets) == 0:
      raise ValueError("No datasets is selected!")
    eval_on(args, model, datasets[0], devices, skip_existing=False)

  else:
    targets = []
    for model_name, (model_dir, runs) in models.items():
      for ds in get_datasets_from_args(args, model_dir):
        for run_dir in runs:
          targets.append((run_dir, ds))

    if len(targets) == 0:
      raise ValueError("No datasets to evaluate on found!")

    for i, (run_dir, dataset) in enumerate(targets):
      if len(targets) > 1:
        logging.info(f"Evaluating on {run_dir} {dataset.get_name()} ({i+1}/{len(targets)})")
      else:
        logging.info(f"Evaluating on {run_dir} {dataset.get_name()}")
      eval_on(args, run_dir, dataset, devices, skip_existing=len(targets) > 1)


if __name__ == '__main__':
  main()
