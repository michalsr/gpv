import argparse
import json

from exp.ours.experiments.trainer_cli import add_train_args, run_train, get_trainer_from_args
from exp.ours.experiments.visual_model_cli import add_image_featurizer_args, get_image_featurizer
from exp.ours.models.gpv1 import GPV1
from exp.ours.image_featurizer.image_featurizer import *
from exp.ours.models.layers import *
from exp.ours.models.model_utils import BackboneParameterExtractor, DetrParameterExtractor
from exp.ours.train.optimizer_builder import *
from exp.ours.util import py_utils
from exp.ours.util.to_params import to_params


def main():
  parser = argparse.ArgumentParser()
  add_train_args(parser, batch_size=120, epochs=40)
  add_image_featurizer_args(parser, vfreeze="none", vmodel="detr_model")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--delay", type=float, default=0.25)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  logging.info("Loading model")
  conf = OmegaConf.load("configs/exp/gpv.yaml").model

  image_featurizer, image_dim = get_image_featurizer(args)

  if args.debug in ["tiny", "small"]:
    conf.co_att.num_layers = 1
    conf.bi_num_attention_heads = 2
    conf.v_intermediate_size = 50

  conf.detr_joiner.detr_dim = image_dim
  model = GPV1(
    conf,
    image_featurizer,
    # unseen_train_mask=args.unseen_train_mask,
  )

  groups = []
  if args.vfreeze != "all":
    groups.append(ParameterGroup(
      BackboneParameterExtractor(),
      group_name="backbone",
      overrides=dict(delay=args.delay, warmup=0.1, lr=1e-5, delay_lr_schedule=False),
    ))
    if args.vmodel == "detr_model":
      groups.append(ParameterGroup(
        DetrParameterExtractor(),
        group_name="detr",
        overrides=dict(delay=args.delay, warmup=0.1, delay_lr_schedule=False),
      ))

  groups.append(ParameterGroup(
      AllParameters(),
      group_name="other",
      overrides=dict(delay=0.0, warmup=0.1, lr=1e-4),
      allow_overlap=True
    ))

  scheduler = DelayedWarmupScheduleBuilder()
  optimizer = AdamWBuilder(
    lr=1e-4,
    weight_decay=1e-4,
    parameter_groups=groups
  )

  print("Optimizer:")
  print(json.dumps(to_params(optimizer, OptimizerBuilder), indent=2))

  get_trainer_from_args(args, logging_ema=0.995, optimizer=optimizer, scheduler=scheduler)


if __name__ == '__main__':
  main()