import argparse
import torch
from omegaconf import OmegaConf

from exp.ours.models.gpv1 import GPV1


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("source")
  parser.add_argument("output")
  args = parser.parse_args()

  print("Loading state...")
  state_dict = torch.load(args.source)
  model_state = state_dict["model"]
  adjusted = {}
  for k, v in model_state.items():
    assert k.startswith("module.")
    adjusted["gpv." + k[len("module."):]] = v

  print("Loading model...")
  conf = OmegaConf.load("configs/exp/gpv.yaml").model
  model = GPV1(conf, initialize_detr=True, freeze_detr=True)

  adjusted["gpv.bert.model.embeddings.position_ids"] = model.gpv.bert.model.embeddings.position_ids.data

  print("Loading state...")
  missing_keys, unexpected = model.load_state_dict(adjusted)
  print(missing_keys)
  print(unexpected)

  print("Saving...")
  torch.save(adjusted, args.output)


if __name__ == '__main__':
  main()