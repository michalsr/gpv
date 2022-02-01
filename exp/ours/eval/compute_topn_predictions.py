
import argparse
import json
import logging

from exp.ours.util.load_model import load_model
import os
from exp.ours.train.optimizer_builder import AllParameters, OptimizerBuilder, \
  DelayedWarmupScheduleBuilder, ParameterGroup, AdamWBuilder
import h5py
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
  Task.WEBQA: 8,
  Task.CLS_IN_CONTEXT: 8,
  Task.CAPTIONING: 30
}

def new_eval_on(task,model):

  tasks = {}  # Use a dictionary to preserve ordering
  for dataset in task:
    print(dataset)
    if dataset == "gpv1":
      tasks.update({x: None for x in GPV1_TASKS})
    elif dataset == "gpv2":
      tasks.update({x: None for x in GPV2_TASKS})
    elif dataset == "non-cls":
      tasks.update({x: None for x in [Task.VQA, Task.CAPTIONING, Task.DETECTION]})
    elif dataset == 'cls':
      print('hello')
      tasks.update({x:None for x in [Task.CLS_IN_CONTEXT]})
    elif dataset == 'det':
      tasks.update({x:None for x in [Task.DETECTION]})
    else:
      tasks[Task(dataset)] = None

  train_datasets = []
  eval_datasets = []
  print(len(tasks))
  for task in tasks:
    #train_datasets.append(TrainerDataset(GpvDataset(task, "train", True,split_txt=''), str(task) + "-tr"))
    eval_datasets.append(TrainerDataset(GpvDataset(task, "test", True), str(task) + "-test"))
  print(len(eval_datasets))
  best_model_key = [
    evaluator.ResultKey("accuracy", dataset_name="cls-val"),
    evaluator.ResultKey("accuracy", dataset_name="cic-val"),
    evaluator.ResultKey("score", dataset_name="vqa-val"),
    evaluator.ResultKey("cider", dataset_name="cap-val"),
    evaluator.ResultKey("accuracy", dataset_name="det-val"),
    evaluator.ResultKey("accuracy", dataset_name="webqa-val"),
  ]
  best_model_key = [x for x in best_model_key if any(x.dataset_name.startswith(str(t)) for t in tasks)]

  


  
  for x in train_datasets:
    if x.dataset.get_task() == Task.CAPTIONING:
      x.eval_sample = 3195
    else:
        x.eval_sample = 3195
  for x in eval_datasets:
    if x.dataset.get_task() == Task.CAPTIONING:
      x.eval_sample = 3195
    else:
      x.eval_sample = 3195

  evaluation = {
    Task.VQA: EvaluationSetup(
      evaluator.VqaEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 10))
    ),
    Task.CAPTIONING: EvaluationSetup(
      evaluator.CaptionEvaluator(per_caption=True),
      dict(beam_search_spec=BeamSearchSpec(1, 30))
    ),
    Task.DETECTION: EvaluationSetup(
      evaluator.LocalizationEvaluator(),
      dict(beam_search_spec=None)
    ),
    Task.CLS: EvaluationSetup(
      evaluator.ClsEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 5), answer_options=CocoCategories())
    ),
    Task.CLS_IN_CONTEXT: EvaluationSetup(
      evaluator.ClsEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 5), answer_options=CocoCategories())
    ),
  }

  train_loader = DataLoaderBuilder(15,0, False,
                                   prefetch_factor=2, persist_workers=False)

  # other_log specifies additional tensorboard logging outputs, we use it to
  # have a second tab with results grouped by train/eval rather than by dataset
  other_log = {}
  evals = [(x, True) for x in train_datasets] + [(x, False) for x in eval_datasets]
  for ds, is_train in evals:
    task = ds.dataset.get_task()
    if task == Task.CAPTIONING:
      metric_name, name = "cider", "cider"
      k = evaluator.ResultKey(metric_name="bleu4", dataset_name=ds.get_name())
      other_log[k] = "bleu4"
    elif task == Task.CLS:
      metric_name, name = "accuracy", "cls"
    elif task == Task.VQA:
      metric_name, name = "score", "vqa"
    elif task == Task.DETECTION:
      metric_name, name = "accuracy", "loc"
    elif task == Task.CLS_IN_CONTEXT:
      metric_name, name = "accuracy", "ident"
    elif task == Task.WEBQA:
      metric_name, name = "accuracy", "webqa"
    else:
      raise RuntimeError()
    name = f"train-evals/{name}" if is_train else f"val-evals/{name}"
    other_log[evaluator.ResultKey(metric_name=metric_name, dataset_name=ds.get_name())] = name
  
  groups = [ParameterGroup(
    AllParameters(),
    group_name="other",
    overrides=dict(delay=0.0, warmup=0.1, lr=0.01),
    allow_overlap=True
  )]
  scheduler = DelayedWarmupScheduleBuilder()
  optimizer = AdamWBuilder(
  lr=0.1,
  weight_decay=1e-4,
  parameter_groups=groups
)
  trainer = Trainer(
    train_datasets,
    eval_datasets,
    evaluation,
    optimizer,

    train_loader=train_loader,

    step_schedule=scheduler,

    save_evaluation_results=True,
    save_prediction_samples=500,
    find_unused_parameters=False,
    train_val_log=list(other_log.items()),
    epochs=1
  )
  eval_examples = [x.dataset.load() for x in eval_datasets]
  print(len(eval_examples),'num eval examples')
  training_examples = [x.dataset.load() for x in train_datasets]
  eval_dir = 'outputs/coco_60_only_test'
  eval_runners = trainer._init_eval(model, training_examples, eval_examples)
  results = trainer._run_eval(model, eval_runners, 0, 0, eval_dir)
  results = {str(k): v for k, v in results.items()}
  print(json.dumps(results, indent=2))


def eval_on(args, run_dir, dataset, devices, skip_existing=False):
  #if args.output_dir:
  output_dir = 'image_contrast_normal_full_3'

 # elif args.output_name:
   # name = f"{dataset.get_name()}--{args.output_name}"
   # eval_dir = join(run_dir, "eval")
    #if not exists(eval_dir):
     # os.mkdir(eval_dir)
    #output_dir = join(eval_dir, name)
  #else:
   # output_dir = None

 # if output_dir is not None:
    #if exists(output_dir):
     # if len(os.listdir(output_dir)) > 0:
        #if skip_existing:
          #logging.info(f"{output_dir} already exists, skipping")
          #return

        #if args.override or py_utils.get_yes_no(f"{output_dir} exists, delete (y/n)?"):
         # logging.info(f"Deleting {output_dir}")
          #rmtree(output_dir)
       # else:
          #logging.info("No override, not stopping")
         # return
    #elif not exists(dirname(output_dir)):
      #raise ValueError(f"Parent folder {dirname(output_dir)} does not exist")
    #else:
      #logging.info(f"Will save to {output_dir}")
  #else:
    #logging.info(f"Not saving the output")

  #if output_dir:
    #if not exists(output_dir):
     # os.mkdir(output_dir)
    #logging.info(f"Saving output to {output_dir}")

  task = dataset.get_task()

  logging.info("Setting up...")
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

  if args.dry_run:
    logging.info("Skipping running the model since this is a dry run")
    return

  output = run(
    run_dir, examples, devices, batch_size, args.num_workers,
    prediction_args, beams_to_keep=beams_to_keep)

  if output_dir is not None:
    logging.info(f"Saving output to {output_dir}")
    #save_gpv_output(output, output_dir)

    config = dict(
      batch_size=batch_size,
      num_workers=args.num_workers,
      predictions_args=prediction_args_to_json(prediction_args),
      dataset=to_params(dataset, Dataset),
      beams_to_keep=beams_to_keep,
      date=datetime.now().strftime("%m%d-%H%M%S"),
    )

    # with open(output_dir + "/config.json", "w") as f:
    #   json.dump(config, f, indent=2)

  if args.eval:
    if isinstance(dataset, OpenSceDataset) and dataset.task == Task.CAPTIONING:
      logging.info("Skip evaluating since no labels OpenSce Captioning")
      return
    else:
      logging.info("Evaluating...")
    evaluator, subsets = get_evaluator(dataset)


  logging.info(f"Saving output to {output_dir}")
  #save_gpv_output(output, output_dir)

  config = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    predictions_args=prediction_args_to_json(prediction_args),
    dataset=to_params(dataset, Dataset),
    date=datetime.now().strftime("%m%d-%H%M%S"),
  )

  # with open(output_dir + "/config.json", "w") as f:
  #   json.dump(config, f, indent=2)

 
  if isinstance(dataset, OpenSceDataset) and dataset.task == Task.CAPTIONING:
    logging.info("Skip evaluating since no labels OpenSce Captioning")
    return
  else:
    logging.info("Evaluating...")
  evaluator, subsets = get_evaluator(dataset)

  results = evaluator.evaluate(examples, output, allow_partial=True, subset_mapping=subsets)

  print(results,'results')
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
  parser.add_argument("--model")
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

  if args.output_dir and args.output_name:
    raise ValueError("Cannot specify output_name and output_dir")
  model_to_eval = 'outputs/auto_select_epoch_4_mil_only'
  models = our_utils.find_models(model_to_eval)
  print(models)
  # if len(models) == 0:
  #   logging.info("No models selected")
  #   return
  run_dir = 'outputs/mil'
  devices = our_utils.get_devices(args.device)
  # if args.output_dir:
  #   models = py_utils.flatten_list(x[1] for x in models.values())
  #   if len(models) > 1:
  #     raise ValueError("Cannot use one output dir if more than one model selected!")
  #model = models[0]

  datasets = get_datasets_from_args(args, model_to_eval)
  eval_on(args,model_to_eval, datasets[0], devices, skip_existing=False)
  #   print(dataset.split_txt)
  #   dataset.change_split("gpv_split")
  #   if len(datasets) > 1:
  #     raise ValueError("Cannot use one output dir if more than one dataset is selected!")
  #   if len(datasets) == 0:
  #     raise ValueError("No datasets is selected!")
  #   #eval_on(args, model, datasets[0], devices, skip_existing=False)
  #   print('new eval')
  #   new_eval_on(dataset,model)
  # else:
  #   targets = []
  #   for model_name, (model_dir, runs) in models.items():
  #     for ds in get_datasets_from_args(args, model_dir):
  #       ds.change_split("gpv_split")
  #       for run_dir in runs:
  #         targets.append((run_dir, ds))

  #   if len(targets) == 0:
  #     raise ValueError("No datasets to evaluate on found!")

  #   for i, (run_dir, dataset) in enumerate(targets):
   
  #     if len(targets) > 1:
  #       logging.info(f"Evaluating on {run_dir} {dataset.get_name()} ({i+1}/{len(targets)})")
  #     else:
  #       logging.info(f"Evaluating on {run_dir} {dataset.get_name()}")
  #     #print(args,run_dir,dataset,devices,args.actual_output_dir,len(targets))
  #     print(ds)
  #new_eval_on(['det'], load_model(run_dir,device=0))


if __name__ == '__main__':
  main()
