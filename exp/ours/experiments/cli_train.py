import logging
from argparse import ArgumentParser

from exp.ours.data.dataset import GpvDataset, InMemoryDataset
from exp.ours.experiments.cli_utils import MarkIfNotDefault
from exp.ours.image_featurizer.image_featurizer import *
from exp.ours.image_featurizer import vinvl_featurizer
from exp.ours.image_featurizer import detr_featurizer
from exp.ours.image_featurizer.vinvl_featurizer import VinvlBackboneImageFeaturizer
from exp.ours.models.layers import DetrBackbone, LayerNorm, BasicBoxEmbedder, \
  NonLinearCoordinateEncoder
from exp.ours.util.our_utils import get_devices
from exp.ours.train.runner import BeamSearchSpec, DataLoaderBuilder
from exp.ours.train import evaluator
from exp.ours.train import optimizer_builder
from exp.ours.train.trainer import TrainerDataset, RunArgs, Trainer, EvaluationSetup

from exp.ours.data.gpv_data import Task, GPV1_TASKS, GPV2_TASKS


def add_image_featurizer_args(parser: ArgumentParser, vfreeze="none", vmodel=None):
  parser.add_argument("--vmodel", default=vmodel)
  parser.add_argument("--vfreeze", default=vfreeze)


def _freeze_detr(vfreeze):
  freeze_extractor = False
  if vfreeze is None or vfreeze == "none":
    freeze_backbone = None
  elif vfreeze == "conv1":
    freeze_backbone = "conv1"
  else:
    freeze_backbone = "all"
    if vfreeze == "backbone":
      freeze_extractor = False
    elif vfreeze == "all":
      freeze_extractor = True
    else:
      raise NotImplementedError()
  return freeze_backbone, freeze_extractor


def get_image_featurizer(args):
  if args.vmodel == "detr_model":
    freeze_backbone, freeze_extractor = _freeze_detr(args.vfreeze)
    dim = 2304
    extractor = detr_featurizer.PretrainedDetrFeaturizer(
      freeze_backbone=freeze_backbone, freeze_extractor=freeze_extractor)
  elif args.vmodel == "faster_rcnn":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("faster-rcnn", "xyxy", BasicBoxEmbedder())
  elif args.vmodel == "dbg":
    dim = 32
    extractor = DebugFeaturizer(4, 32)
  elif args.vmodel == "vinvl-precomputed":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2054
    extractor = vinvl_featurizer.VinVLPrecomputedFeatures()
  elif args.vmodel == "dbg-hdf5":
    dim = 2048
    extractor = Hdf5FeatureExtractor("dbg")
  elif args.vmodel == "vinvl":
    if args.vfreeze != "all":
      raise ValueError()
    # dim = 2048 + 5
    # extractor = Hdf5FeatureExtractor("vinvl", "xyxy", BasicBoxEmbedder())
    dim = 2048
    extractor = Hdf5FeatureExtractor("vinvl")
  elif args.vmodel == "vinvl-r50c4-4setvg":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("vinvl-r50c4-4setvg", "xyxy", BasicBoxEmbedder())
  elif args.vmodel == "vinvl-r50c4-4setvg-rboxes":
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("vinvl-r50c4-4setvg-rboxes", box_embedder=BasicBoxEmbedder())
  elif args.vmodel == "vinvl-r50c4-4setvg-rboxes-all-image":
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("vinvl-r50c4-4setvg-rboxes-all-image", box_embedder=BasicBoxEmbedder(),
                                     all_image_box=True, all_image_prior=-10000)
  elif args.vmodel == "vinvl-r50c4-4setvg-rboxes-bk":
    dim = 2048 + 5
    extractor = VinvlBackboneImageFeaturizer("vinvl", "R50C4_4setsvg", BasicBoxEmbedder(), args.vfreeze)
  elif args.vmodel == "vinvl-r50c4-4setvg-bk":
    dim = 2048 + 5
    extractor = VinvlBackboneImageFeaturizer("vinvl-r50c4-5setvg", "R50C4_4setsvg", BasicBoxEmbedder(), args.vfreeze)
  elif args.vmodel == "vinvl-r50c4-4setvg-rboxes-bk-aimg":
    dim = 2048 + 5
    extractor = VinvlBackboneImageFeaturizer(
      "vinvl", "R50C4_4setsvg", BasicBoxEmbedder(), args.vfreeze, True)
  elif args.vmodel == "vinvl_backbone":
    dim = 2048 + 5
    extractor = VinvlBackboneImageFeaturizer("vinvl", "release", BasicBoxEmbedder(), args.vfreeze)
  elif args.vmodel in {"detr_boxes", "vinvl_boxes"}:
    dim = 2048 + 4*7
    extractor = FromPrecomputedBoxes(
      {"detr_boxes": "detr-coco-sce", "vinvl_boxes": "vinvl-boxes"}[args.vmodel],
      DetrBackbone(freeze=args.vfreeze),
      feature_extractor=BoxEmbedFeatureExtractor(
        box_coordinate_embed=NonLinearCoordinateEncoder([0.1, 0.05, 0.02]),
        post_rio=LayerNorm(),
      ),
      include_all_image_box=True,
      horizontal_flip=0.5,
      preload_bboxes=True,
    )
  else:
    raise NotImplementedError(args.vmodel)
  return extractor, dim


def add_train_args(parser: ArgumentParser, batch_size=32, num_workers=6,
                   epochs=40, tasks="all", clip_grad_norm=0.1):
  parser.add_argument("--tasks", nargs="+", default=tasks)

  # Performance args
  parser.add_argument("--device", nargs="+", default=None)
  parser.add_argument("--dist_port", default=None, type=int)
  parser.add_argument("--grad_accumulation", type=int, default=1)
  parser.add_argument("--force_one_worker", action="store_true")
  parser.add_argument("--nopin_memory", action="store_true")
  parser.add_argument("--num_workers", default=num_workers, type=int, action=MarkIfNotDefault)

  parser.add_argument("--clip_grad_norm", default=clip_grad_norm, type=float)

  # Other training args
  parser.add_argument("--batch_size", default=batch_size, type=int, action=MarkIfNotDefault)
  parser.add_argument("--epochs", default=epochs, type=int)
  parser.add_argument("--debug", choices=["tiny", "small", "med", "large"], default=None)

  # Run format args
  parser.add_argument("--eval_start", action="store_true")
  parser.add_argument("--override", action="store_true")
  parser.add_argument("--output_dir")


# TODO move optimizer and warmup out of this method
def run_train(args, model, logging_ema=0.99, sync_monitor=True,
              epoch_end_hook=None, groups=None, vision_regex=None,
              find_unused_parameters=True,
              optimizer=None, scheduler=None
              ):
  if args.tasks == ["all"] or args.tasks == "all":
    tasks = list(Task)
  elif args.tasks == ["gpv1"]:
    tasks = GPV1_TASKS
  elif args.tasks == ["gpv2"]:
    tasks = GPV2_TASKS
  else:
    tasks = [Task(x) for x in args.tasks]
    if len(set(tasks)) != len(args.tasks):
      raise ValueError("Given the same task multiple times")

  logging.info(f"Selected tasks: {[str(x) for x in tasks]}")
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

  train_dataset = [
    TrainerDataset(
      GpvDataset(task, "train", True),
      logging_name=str(task) + "-tr",
    ) for task in tasks
  ]

  eval_datasets = [
    TrainerDataset(
      GpvDataset(task, "val", True),
      logging_name=str(task) + "-val",
    ) for task in tasks
  ]

  if args.debug == "tiny":
    for x in train_dataset:
      x.dataset.sample = 5
      x.eval_sample = 4
    for x in eval_datasets:
      x.dataset.sample = 5
      x.eval_sample = 4

  elif args.debug == "small":
    for x in train_dataset:
      x.dataset.sample = 120
      x.eval_sample = 30
    for x in eval_datasets:
      x.dataset.sample = 120
      x.eval_sample = 30

  elif args.debug == "med":
    for x in train_dataset:
      x.dataset.sample = 2000
      x.eval_sample = 500
    for x in eval_datasets:
      x.dataset.sample = 2000
      x.eval_sample = 500

  elif args.debug == "large":
    for x in train_dataset:
      x.dataset.sample = 10000
      x.eval_sample = 2000
    for x in eval_datasets:
      x.eval_sample = 2000

  else:
    for x in train_dataset:
      if x.dataset.get_task() == Task.CAPTIONING:
        x.eval_sample = 5000
      else:
        x.eval_sample = 8000
    for x in eval_datasets:
      if x.dataset.get_task() == Task.CAPTIONING:
        x.eval_sample = 8000
      else:
        x.eval_sample = 12000

  best_model_key = [
    evaluator.ResultKey("accuracy", dataset_name="cls-val"),
    evaluator.ResultKey("score", dataset_name="vqa-val"),
    evaluator.ResultKey("cider", dataset_name="cap-val"),
    evaluator.ResultKey("AP", dataset_name="det-val"),
  ]
  best_model_key = [x for x in best_model_key if any(x.dataset_name.startswith(str(t)) for t in tasks)]

  evaluation = {
    Task.VQA: EvaluationSetup(
      evaluator.VqaEvaluator(),
      dict(allennlp_spec=BeamSearchSpec(1, 8))
    ),
    Task.CAPTIONING: EvaluationSetup(
      evaluator.CaptionEvaluator(per_caption=True),
      dict(allennlp_spec=BeamSearchSpec(1, 30))
    ),
    Task.DETECTION: EvaluationSetup(
      evaluator.DetectionEvaluator(),
      dict(predict_text=False)
    ),
    Task.CLS: EvaluationSetup(
      evaluator.ClsEvaluator(),
      dict(allennlp_spec=BeamSearchSpec(1, 4))
    ),
    Task.CLS_IN_CONTEXT: EvaluationSetup(
      evaluator.ClsEvaluator(),
      dict(allennlp_spec=BeamSearchSpec(1, 4))
    )
  }

  if scheduler is None:
    raise ValueError()

  train_loader = DataLoaderBuilder(batch_size, num_workers, not args.nopin_memory,
                                   prefetch_factor=2, persist_workers=num_workers > 0)

  other_log = {}
  evals = [(x, True) for x in train_dataset] + [(x, False) for x in eval_datasets]
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
    else:
      raise RuntimeError()
    name = f"train-evals/{name}" if is_train else f"val-evals/{name}"
    other_log[evaluator.ResultKey(metric_name=metric_name, dataset_name=ds.get_name())] = name

  trainer = Trainer(
    train_dataset,
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
    tb_log_intervals=10,
    clip_grad_norm_re=vision_regex,
    epoch_end_hook=epoch_end_hook,
    sort_train=False,
    checkpoint=True,
    sync_monitor=sync_monitor,
    eval_at_start=args.eval_start,
    loss_logging_ema=logging_ema,
    monitor_ema=logging_ema,
  )

  devices = RunArgs.build(get_devices(args.device), args.force_one_worker, args.grad_accumulation)

  trainer.train(model, args.output_dir, devices, override=args.override)
