import json
import logging
import os
from copy import deepcopy

import torch.utils.data
from transformers import AutoConfig

from exp.ours.data.opensce import OpenSceDataset
from exp.ours.data.webqa import WebQaDataset, WebQaNoTemmplatesDataset
from exp.ours.data.webqa_templates import WebQaQueryGenerator, TemplateWebQueryGenerator
from exp.ours.experiments.visual_model_cli import add_image_featurizer_args, get_image_featurizer
from exp.ours.models.layers import *
from exp.ours.models.losses import *
from exp.ours.models.t5_gpv import T5GPV
from exp.ours.models.t5_gpv_per_box import T5GpvPerBox
from exp.ours.train.evaluator import ResultKey
from exp.ours.train.optimizer_builder import AllParameters, OptimizerBuilder, \
  DelayedWarmupScheduleBuilder, ParameterGroup, AdamWBuilder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from exp.ours.experiments.trainer_cli import *
from exp.ours.data.dataset import Task
from exp.ours.util import py_utils


def main():
  parser = ArgumentParser()
  parser.add_argument("--model", choices=["t5-small", "t5-base", "t5-large"], default=None)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--webqa_sample", type=float, default=1.0)
  parser.add_argument("--webqa_subset",  default=None)
  parser.add_argument("--query_box",  default="always")
  parser.add_argument("--find_unused", action="store_true")
  parser.add_argument("--init_from")
  parser.add_argument("--train_from")
  parser.add_argument("--vwarmup", type=float, default=0.1)
  parser.add_argument("--sce", action="store_true")
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--vlr", type=float)
  parser.add_argument("--delay", type=float, default=0.0)

  add_image_featurizer_args(parser, vfreeze="all", vmodel="vinvl")
  add_train_args(
    parser, tasks=[str(Task.CAPTIONING)], epochs=4,
    clip_grad_norm=None, num_workers=4, batch_size=60)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.model is None:
    if args.debug in ["tiny", "small"] and args.init_from is None:
      args.model = "t5-small"
    else:
      args.model = "t5-base"

  image_featurizer, image_dim = get_image_featurizer(args)

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model

  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels'])

  # Current best
  model = T5GpvPerBox(
    args.model,
    loss=BasicGPVLoss(localization_loss),
    image_feature_extractor=image_featurizer,
    image_joiner=Linear(image_dim, t5_dim),
    pre_tokenize=True,
    query_box=None if args.query_box == "none" else args.query_box,
    all_lower_case=True,
    webqa_templates=TemplateWebQueryGenerator(use_commands=True),
    initialize_from=args.init_from,
    contrast_query="other",
    convert_to_relevance="raw",
    combine_with_objectness="multiply",
    embed_objectness_score=False,
  )

  # model = T5GPV(
  #   args.model,
  #   loss=BasicGPVLoss(localization_loss),
  #   image_feature_extractor=image_featurizer,
  #   image_joiner=Linear(image_dim, t5_dim),
  #   pre_tokenize=True,
  #   image_relevance=SumWithObjectness(t5_dim, objectness_factor=True),
  #   query_box=None if args.query_box == "none" else args.query_box,
  #   all_lower_case=True,
  #   webqa_templates=TemplateWebQueryGenerator(use_commands=True),
  #   initialize_from=args.init_from,
  # )

  groups = [ParameterGroup(
    AllParameters(),
    group_name="other",
    overrides=dict(delay=0.0, warmup=0.1, lr=args.lr),
    allow_overlap=True
  )]

  scheduler = DelayedWarmupScheduleBuilder()
  optimizer = AdamWBuilder(
    lr=args.lr,
    weight_decay=args.weight_decay,
    parameter_groups=groups
  )

  print("Optimizer:")
  print(json.dumps(to_params(optimizer, OptimizerBuilder), indent=2))

  trainer: Trainer = get_trainer_from_args(
    args, logging_ema=0.995, find_unused_parameters=True,
    optimizer=optimizer, scheduler=scheduler
  )

  if args.webqa_subset is not None:

    if args.webqa_subset == "notemplates-5/6":
      qtypes = tuple("5a 5n 5v 6a 6v".split())
      webqa_train = WebQaNoTemmplatesDataset("train", 100 if args.debug else None, qtypes)
      webqa_val = WebQaNoTemmplatesDataset("val", 100 if args.debug else None, qtypes)
    elif args.webqa_subset == "notemplates-3/4":
      qtypes = tuple("3a 3n 3v 4a 4v".split())
      webqa_train = WebQaNoTemmplatesDataset("train", 100 if args.debug else None, qtypes)
      webqa_val = WebQaNoTemmplatesDataset("val", 100 if args.debug else None, qtypes)
    else:
      qtypes = args.webqa_subset
      qtypes = WebQaDataset.QTYPES_NAME_TO_TYPES.get(qtypes, (qtypes,))
      webqa_train = WebQaDataset("val" if args.debug else "train",
                                 100 if args.debug else None, qtypes)
      webqa_val = WebQaDataset("val", 100 if args.debug else None, qtypes)

    webqq_eval = EvaluationSetup(
      evaluator.WebQaEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 5))
    )

    trainer.train_datasets.append(TrainerDataset(
      webqa_train, "webqa-tr",
      train_sample=args.webqa_sample,
      eval_sample=50 if args.debug else 3000, eval_setup=webqq_eval
    ))
    trainer.eval_datasets.append(TrainerDataset(
      webqa_val, "webqa-val", eval_sample=50 if args.debug else 12000, eval_setup=webqq_eval))
    trainer.best_model_key.append(ResultKey("accuracy", dataset_name="webqa-val"))

  trainer.stratify = True
  trainer.eval_loader = deepcopy(trainer.train_loader)

  if not args.sce:
    for ds in trainer.train_datasets:
      ds.dataset.gpv_split = False
    for ds in trainer.eval_datasets:
      ds.dataset.gpv_split = False

  trainer.train_loader.persist_workers = False
  trainer.eval_loader.persist_workers = False
  trainer.find_unused_parameters = args.find_unused

  run_trainer_from_args(trainer, model, args)


if __name__ == '__main__':
  main()
