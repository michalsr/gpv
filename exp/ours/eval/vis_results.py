
import argparse
import json
import logging

from exp.ours.util.load_model import load_model
import os
from exp.ours.train.optimizer_builder import AllParameters, OptimizerBuilder, \
  DelayedWarmupScheduleBuilder, ParameterGroup, AdamWBuilder
import h5py
import exp.ours.util.io as io
from exp.ours.train import evaluator
from exp.ours.train.evaluator import ResultKey, CaptionEvaluator, Evaluator
from exp.ours.boosting import SceUnseenCategories, OpenSceUnseenCategories
from exp.ours.data.gpv import GpvDataset
from exp.ours.data.opensce import OpenSceDataset
from exp.ours.eval.eval_predictions import get_evaluator, cache_evaluation
from exp.ours.train.runner import BeamSearchSpec
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from exp.ours.data.gpv import GpvDataset, CocoCategories
from datetime import datetime
from os.path import join, exists, dirname
from shutil import rmtree
from exp.ours.train.runner import BeamSearchSpec, DataLoaderBuilder
from allennlp.common import Registrable
from dataclasses import dataclass
from exp.ours.data.dataset import GPV1_TASKS, GPV2_TASKS, Task
from exp.ours.experiments.datasets_cli import add_dataset_args, get_datasets_from_args
from exp.ours.train.evaluator import ResultKey
from exp.ours.util import our_utils, py_utils, image_utils
from exp.ours.data.dataset import Dataset, Task, InMemoryDataset
from exp.ours.train.runner import BeamSearchSpec, save_gpv_output, \
  run, prediction_args_to_json
from exp.ours.util.to_params import to_params
from exp.ours.train.trainer import TrainerDataset, RunArgs, Trainer, EvaluationSetup
import exp.ours.experiments.configure_train_datasets as configure_dataset 






def run_eval(args, run_dir, dataset):
    loggingng.info("Setting up...")
    task = dataset.get_task()
    examples = dataset.load()
    do_rerank = False
  if args.rank_answer_options == "always":
    do_rerank = task in {Task.CLS, Task.CLS_IN_CONTEXT}
  elif args.rank_answer_options == "never":
    do_rerank = False
  elif args.rank_answer_options == "non-webqa":
    if "webqa" not in dataset.get_name():
      do_rerank = task in {Task.CLS, Task.CLS_IN_CONTEXT}
  else:
    raise NotImplementedError(args.rank_answer_options)

  prediction_args = {}
  beams_to_keep = args.beams_to_keep
  batch_size = args.batch_size

  if task in {Task.CLS, Task.CLS_IN_CONTEXT, Task.WEBQA} and args.cls_mask != "none":
    answer_options = dataset.get_answer_options(args.cls_mask == "synonyms")
    prediction_args["answer_options"] = answer_options
    logging.info(f"Using classification mask for {len(answer_options)} words")

  if task in {Task.CLS, Task.CLS_IN_CONTEXT, Task.WEBQA}:
    logging.info("Classification so keeping 20 beams")
    beams_to_keep = 20

  if do_rerank and prediction_args.get("answer_options"):
    logging.info(f"Re-ranking answer options")
    logging.info(f"Reducing batch size to 5")
    batch_size = 5
    prediction_args["rerank_answer_options"] = True
  else:
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
    prediction_args["beam_search_spec"] = bs

  if args.boost_unseen:
    if isinstance(dataset, GpvDataset):
      prediction_args["mask"] = SceUnseenCategories(task, args.boost_unseen, args.boost_syn)
    elif isinstance(dataset, OpenSceDataset):
      if dataset.task == Task.CLS:
        prediction_args["mask"] = OpenSceUnseenCategories(task, args.boost_unseen, args.boost_syn)
      else:
        # prediction_args["mask"] = WebQaAnswersBoost(args.boost_unseen)
        prediction_args["mask"] = WebQaAnswersBoost(args.boost_unseen)
    else:
      raise NotImplementedError()
    output = run(
    run_dir, examples, devices, batch_size, args.num_workers,
    prediction_args, beams_to_keep=beams_to_keep)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name",type=str,default=None)
  parser.add_argument("--data_file",type=str,default=None)
  parser.add_argument("--file_prefix",type=str,default=None)
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
  parser.add_argument("--rank_answer_options", default="non-webqa", choices=["never", "always", "non-webqa"])
  parser.add_argument("--nms", type=float, default=None)
  parser.add_argument("--actual_output_dir",type=str,default='outputs/new_1')
  args = parser.parse_args()

  py_utils.add_stdout_logger()


  model_to_eval = args.model_name
  models = our_utils.find_models(model_to_eval)


  run_dir = 'outputs/mil'
  devices = our_utils.get_devices(args.device)
  single_image_data = io.load_json_object(f'{args.file_prefix}/{args.data_file}')

  datasets = get_datasets_from_args(args, model_to_eval)
  eval_on(args,model_to_eval, datasets[0], devices, skip_existing=False)
 

if __name__ == '__main__':
  main()
