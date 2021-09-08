import json
import os
from copy import deepcopy

import torch.utils.data
from transformers import AutoConfig

from exp.ours.data.coco_segmentation import CocoSegmentationDataset
from exp.ours.models.layers import *
from exp.ours.models.losses import *
from exp.ours.train.evaluator import ResultKey, SegmentationEvaluator
from exp.ours.train.optimizer_builder import AllParameters, OptimizerBuilder, \
  DelayedWarmupScheduleBuilder, ParameterGroup, AdamWBuilder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from exp.ours.experiments.trainer_cli import *
from exp.ours.data.dataset import Task
from exp.ours.models.t5_gpv import T5GPV
from exp.ours.util import py_utils


def main():
  parser = ArgumentParser()
  parser.add_argument("--model", choices=["t5-small", "t5-base", "t5-large"], default=None)
  parser.add_argument("--output_dir", default=None)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--epochs", type=int, default=8)
  parser.add_argument("--warmup", type=float, default=0.1)
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--device", nargs="+", default=None)
  parser.add_argument("--num_workers", default=4, type=int)
  parser.add_argument("--batch_size", default=60, type=int)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.model is None:
    if args.debug in ["tiny", "small"]:
      args.model = "t5-small"
    else:
      args.model = "t5-base"

  image_featurizer = Hdf5FeatureExtractor("coco/vinvl", box_embedder=BasicBoxEmbedder())
  image_dim = 2048 + 5

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model

  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels'])
  loss = BasicGPVLoss(localization_loss)
  model = T5GPV(
    args.model,
    loss=loss,
    image_feature_extractor=image_featurizer,
    image_joiner=Linear(image_dim, t5_dim),
    pre_tokenize=True,
    image_relevance=SumWithObjectness(t5_dim, objectness_factor=True),
    query_box="always",
    all_lower_case=True,
    webqa_templates=None
  )

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

  if args.debug is None:
    tr_sample, eval_sample = None, None
  else:
    tr_sample, eval_sample = 500, 100

  trainer = Trainer(
    train_datasets=[TrainerDataset(
      CocoSegmentationDataset("train", tr_sample),
      eval_sample=None if args.debug else 10000,
      logging_name="seg-tr"
    )],
    eval_datasets=[TrainerDataset(
      CocoSegmentationDataset("val", eval_sample),
      logging_name="seg-val",
      eval_sample=None if args.debug else 15000,
    )],
    evaluation={
      Task.SEGMENTATION: EvaluationSetup(
        SegmentationEvaluator(),
        {}
      )
    },
    train_loader=DataLoaderBuilder(
      batch_size=args.batch_size,
      num_workers=args.num_workers
    ),
    optimizer=optimizer,
    step_schedule=scheduler,
    epochs=args.epochs
  )

  devices = RunArgs.build(get_devices(args.device), False)
  trainer.train(model, args.output_dir, devices, override=False)


if __name__ == '__main__':
  main()
