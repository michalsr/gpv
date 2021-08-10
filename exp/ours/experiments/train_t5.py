import json
import os

from transformers import AutoConfig

from exp.ours.data.webqa import Web80QaDataset
from exp.ours.experiments.visual_model_cli import add_image_featurizer_args, get_image_featurizer
from exp.ours.models.layers import *
from exp.ours.models.losses import *
from exp.ours.models.model_utils import BackboneParameterExtractor
from exp.ours.train.evaluator import ResultKey
from exp.ours.train.optimizer_builder import AllParameters, OptimizerBuilder, \
  DelayedWarmupScheduleBuilder, ParameterGroup, AdamWBuilder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from exp.ours.experiments.trainer_cli import *
from exp.ours.data.gpv_data import Task
from exp.ours.models.t5_gpv import T5GPV
from exp.ours.util import py_utils


def main():
  parser = ArgumentParser()
  parser.add_argument("--model", choices=["t5-small", "t5-base", "t5-large"], default=None)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--vlr", type=float)
  parser.add_argument("--vwarmup", type=float, default=0.1)
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--delay", type=float, default=0.0)

  add_image_featurizer_args(parser, vfreeze="all", vmodel="detr_boxes")
  add_train_args(
    parser, tasks=[str(Task.CAPTIONING)], epochs=4,
    clip_grad_norm=None, num_workers=4, batch_size=60)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.model is None:
    if args.debug in ["tiny", "small"]:
      args.model = "t5-small"
    else:
      args.model = "t5-base"

  image_featurizer, image_dim = get_image_featurizer(args)

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model

  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels', 'boxes'])
  model = T5GPV(
    args.model,
    loss=BasicGPVLoss(1, 1, 1, 1, 1, localization_loss, False),
    image_feature_extractor=image_featurizer,
    image_joiner=Linear(image_dim, t5_dim),
    pre_tokenize=True,
    image_relevance=SumWithObjectness(t5_dim, objectness_factor=True),
    query_box="always",
    all_lower_case=False
  )

  groups = []
  if args.vfreeze not in {"all", "backbone"}:
    groups.append(ParameterGroup(
      BackboneParameterExtractor(),
      group_name="backbone",
      overrides=dict(delay=args.delay, warmup=args.vwarmup, lr=args.vlr),
    ))

  groups.append(ParameterGroup(
    AllParameters(),
    group_name="other",
    overrides=dict(delay=0.0, warmup=0.1, lr=args.lr),
    allow_overlap=True
  ))

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

  # Add the WebQaDataset we want to experiment with
  trainer.train_datasets.append(TrainerDataset(
    Web80QaDataset(100 if args.debug else None, "train"), "webqa-tr",
    eval_sample=50 if args.debug else 3000
  ))
  trainer.eval_datasets.append(TrainerDataset(
    Web80QaDataset(100 if args.debug else None, "val"), "webqa-val",
    eval_sample=50 if args.debug else 5000
  ))
  trainer.best_model_key.append(ResultKey("accuracy", dataset_name="webqa-val"))
  trainer.evaluation[Task.WEBQA] = EvaluationSetup(
    evaluator.WebQaEvaluator(),
    dict(beam_search_spec=BeamSearchSpec(1, 5), answer_options=WebQa80Answers())
  )

  run_trainer_from_args(trainer, model, args)


if __name__ == '__main__':
  main()
