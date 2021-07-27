import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse

from exp.ours.util import our_utils, py_utils
from exp.ours.util.our_utils import get_devices
from exp.ours.train.trainer import Trainer, RunArgs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("output_dir")
  parser.add_argument("--device", nargs="+", default=None)
  parser.add_argument("--num_workers", default=None, type=int)
  parser.add_argument("--force_one_worker", action="store_true")
  parser.add_argument("--nosave", action="store_true")

  args = parser.parse_args()
  py_utils.add_stdout_logger()

  our_utils.import_all()
  device = RunArgs.build(
    get_devices(args.device), args.force_one_worker, num_workers=args.num_workers)
  Trainer.train_another_model(args.output_dir, device, save=not args.nosave)


if __name__ == '__main__':
  main()