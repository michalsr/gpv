import argparse
import json
import logging
from os.path import join, exists

from exp.ours import file_paths
from exp.ours.data.dataset import Task
from exp.ours.data.opensce import OpenSceDataset
from exp.ours.models import model_utils
from exp.ours.train.runner import load_gpv_predictions
from exp.ours.util import our_utils, py_utils
from utils.io import load_json_object


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  args = parser.parse_args()

  models = our_utils.find_models(args.model)
  if len(models) == 0:
    logging.info("No models found")
    return

  dataset = OpenSceDataset(Task.CAPTIONING, "val")
  instances = dataset.load()
  cap_name = dataset.get_name()
  eval_name = cap_name + "--basic"
  image_info = load_json_object(file_paths.NOCAPS_VAL_IMAGE_INFO)['images']
  image_id_to_id = {info['open_images_id']:info['id'] for info in image_info}

  for model_name, (model_dir, runs) in models.items():
    for run in runs:
      target_dir = join(run, "eval", eval_name)
      if not exists(target_dir):
        logging.info(f"Skip model {model_name}, missing {target_dir}")
        continue

      target_file = join(target_dir, "val-submission.json")
      if exists(target_file):
        logging.info(f"Skip model {model_name}, already has submission file {target_file}")
        continue

      logging.info(f"Building submission file {target_file}")
      out = []
      predictions = load_gpv_predictions(target_dir)
      for instance in instances:
        out.append(dict(
          image_id=image_id_to_id[instance.image_id.split("/")[-1].split(".")[0]],
          caption=predictions[instance.get_gpv_id()].text[0]
        ))
      with open(target_file, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
  py_utils.add_stdout_logger()
  main()