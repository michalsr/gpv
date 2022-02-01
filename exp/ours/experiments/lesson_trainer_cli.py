from argparse import ArgumentParser

from exp.ours.data.dataset import GPV1_TASKS, GPV2_TASKS, Task
from exp.ours.data.gpv import GpvDataset, CocoCategories
from exp.ours.image_featurizer.image_featurizer import *
from exp.ours.util.our_utils import get_devices
from exp.ours.train.runner import BeamSearchSpec, DataLoaderBuilder
from exp.ours.train import evaluator
from exp.ours.train.lesson_trainer import TrainerDataset, RunArgs, Trainer, EvaluationSetup

from exp.ours.util.py_utils import MarkIfNotDefault


def add_train_args(parser: ArgumentParser, batch_size=32, num_workers=6,
                   epochs=40, tasks="all", clip_grad_norm=None):
  parser.add_argument("--task", nargs="+", default=tasks)

  # Performance args
  parser.add_argument("--device", nargs="+", default=None)
  parser.add_argument("--dist_port", default=None, type=int)
  parser.add_argument("--grad_accumulation", type=int, default=1)
  parser.add_argument("--force_one_worker", action="store_true")
  parser.add_argument("--nopin_memory", action="store_true")
  parser.add_argument("--num_workers", default=num_workers, type=int, action=MarkIfNotDefault)


  # Other training args
  parser.add_argument("--clip_grad_norm", default=clip_grad_norm, type=float)
  parser.add_argument("--batch_size", default=batch_size, type=int, action=MarkIfNotDefault)
  parser.add_argument("--epochs", default=epochs, type=int)
  parser.add_argument("--debug", choices=["tiny", "small", "med", "large"], default=None)

  parser.add_argument("--split_txt",default="held_out_all",type=str)

  # Run format args
  parser.add_argument("--eval_start", action="store_true")
  parser.add_argument("--override", action="store_true")
  parser.add_argument("--output_dir")


def run_train(args, model, **kwargs):
  trainer = get_trainer_from_args(args, **kwargs)
  run_trainer_from_args(trainer, model, args)


def get_trainer_from_args(
    args, optimizer, logging_ema=0.99, sync_monitor=True, vision_regex=None,
    find_unused_parameters=True, scheduler=None
) -> Trainer:
  batch_size, num_workers = args.batch_size, args.num_workers

  if args.debug:
    dbg_batch_size, dbg_num_workers = {
      "tiny": (2, 0),
      "small": (8, 0),
      "med": (24, 4),
      "large": (60, 4),
    }[args.debug]
    if not hasattr(args, "batch_size_not_default"):
      batch_size = dbg_batch_size
    if not hasattr(args, "num_workers_not_default"):
      num_workers = dbg_num_workers

  logging.info(f"batch size={batch_size}")
  logging.info(f"num_workers={num_workers}")
  logging.info(f"lr={args.lr}")
  if args.grad_accumulation != 1:
    logging.info(f"grad acc={args.grad_accumulation}")

  tasks = {}  # Use a dictionary to preserve ordering
  for dataset in args.task:
    if dataset == "gpv1":
      tasks.update({x: None for x in GPV1_TASKS})
    elif dataset == "gpv2":
      tasks.update({x: None for x in GPV2_TASKS})
    elif dataset == "non-cls":
      tasks.update({x: None for x in [Task.VQA, Task.CAPTIONING, Task.DETECTION]})
    elif dataset == 'lesson':
      continue 
    elif dataset == 'webqa':
      continue
    elif dataset == 'img-contrast' or dataset == 'mil':
      continue
    else:
      tasks[Task(dataset)] = None

  train_datasets = []
  eval_datasets = []
  for task in tasks:
    train_datasets.append(TrainerDataset(GpvDataset(task, "train", True,split_txt=args.split_txt), str(task) + "-tr"))
    eval_datasets.append(TrainerDataset(GpvDataset(task, "val", True,split_txt=args.split_txt), str(task) + "-val"))

  best_model_key = [
    evaluator.ResultKey("accuracy", dataset_name="cls-val"),
    evaluator.ResultKey("accuracy", dataset_name="cic-val"),
    evaluator.ResultKey("score", dataset_name="vqa-val"),
    evaluator.ResultKey("cider", dataset_name="cap-val"),
    evaluator.ResultKey("AP", dataset_name="det-val"),
    evaluator.ResultKey("accuracy", dataset_name="webqa-val"),
  ]
  best_model_key = [x for x in best_model_key if any(x.dataset_name.startswith(str(t)) for t in tasks)]

  if args.debug == "tiny":
    for x in train_datasets:
      x.dataset.sample = 5
      x.eval_sample = 4
    for x in eval_datasets:
      x.dataset.sample = 5
      x.eval_sample = 4

  elif args.debug == "small":
    for x in train_datasets:
      x.dataset.sample = 120
      x.eval_sample = 30
    for x in eval_datasets:
      x.dataset.sample = 120
      x.eval_sample = 30

  elif args.debug == "med":
    for x in train_datasets:
      x.dataset.sample = 2000
      x.eval_sample = 500
    for x in eval_datasets:
      x.dataset.sample = 2000
      x.eval_sample = 500

  elif args.debug == "large":
    for x in train_datasets:
      x.dataset.sample = 10000
      x.eval_sample = 2000
    for x in eval_datasets:
      x.eval_sample = 2000

  else:
    for x in train_datasets:
      if x.dataset.get_task() == Task.CAPTIONING:
        x.eval_sample = 5000
      elif x.dataset.get_task() == Task.WEBQA:
        x.eval_sample = 1314
      else:
        x.eval_sample = 8000
   
    for x in eval_datasets:
      #print(x.dataset.get_task())
      if x.dataset.get_task() == Task.CAPTIONING:
        x.eval_sample = 8000
      elif x.dataset.get_task() == Task.WEBQA:
        x.eval_sample = 1314
      else:
        x.eval_sample = 39426
      x.eval_sample=39426

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

  train_loader = DataLoaderBuilder(batch_size, num_workers, not args.nopin_memory,
                                   prefetch_factor=2, persist_workers=num_workers > 0)

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
      metric_name, name = "AP", "loc"
    elif task == Task.CLS_IN_CONTEXT:
      metric_name, name = "accuracy", "ident"
    elif task == Task.WEBQA:
      metric_name, name = "accuracy", "webqa"
    else:
      raise RuntimeError()
    name = f"train-evals/{name}" if is_train else f"val-evals/{name}"
    other_log[evaluator.ResultKey(metric_name=metric_name, dataset_name=ds.get_name())] = name

  trainer = Trainer(
    train_datasets,
    eval_datasets,
    evaluation,
    optimizer,

    train_loader=train_loader,

    step_schedule=scheduler,

    save_evaluation_results=True,
    save_prediction_samples=500,
    find_unused_parameters=find_unused_parameters,

    train_val_log=list(other_log.items()),

    epochs=args.epochs,
    best_model_key=best_model_key,
    clip_grad_norm=args.clip_grad_norm,
    tb_log_intervals=20,
    clip_grad_norm_re=vision_regex,
    checkpoint=True,
    sync_monitor=sync_monitor,
    eval_at_start=args.eval_start,
    loss_logging_ema=logging_ema,
    monitor_ema=logging_ema,
  )

  return trainer


def run_trainer_from_args(trainer, model, args):
  devices = RunArgs.build(get_devices(args.device), args.force_one_worker, args.grad_accumulation)
  trainer.train(model, args.output_dir, devices, override=args.override)
