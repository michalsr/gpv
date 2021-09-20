"""This script demos how to use our model to re-rank a list of answer options"""
import argparse

import torch.cuda

from exp.ours.data.dataset import Task
from exp.ours.data.gpv_example import GPVExample
from exp.ours.models.model import GPVExampleOutput
from exp.ours.util import our_utils
from exp.ours.util.load_model import load_model
from exp.ours.util.our_utils import select_run_dir


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model", help="Model directory to run with")
  parser.add_argument("image_id",
                      help="Image id to use, it has to be something the model can "
                           "locate in its HDF5 feature files")
  parser.add_argument("query", help="Query to give the model")
  parser.add_argument("answer_options", nargs="+")
  args = parser.parse_args()

  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  run = select_run_dir(args.model)
  print(f"Loading model from {run}...")
  model = load_model(run, device=device)

  try:
    # For coco-images the image_id is an integer
    image_id = int(args.image_id)
  except ValueError:
    image_id = args.image_id

  # This tell the model what the answer options are, and to do re-ranking
  # It only needs to be called once if you are running on multiple batches
  model.set_prediction_args(
    answer_options=args.answer_options,
    rerank_answer_options=True
  )

  print("Done, setting up inputs...")
  query_input = GPVExample(
    id="", task=None,
    image_id=image_id,
    query=args.query,
  )
  input_batch = [query_input]
  input_batch = [model.preprocess_example(x) for x in input_batch]
  features = model.get_collate()(input_batch)
  features = our_utils.to_device(features, device)

  print("Starting prediction...")
  with torch.no_grad():
    output: GPVExampleOutput = model.predict(**features)[0]

  print("Done, printing outputs")
  for text, prob in zip(output.text, output.text_logprobs):
    print(f"{text}: {prob:.5g}")


if __name__ == '__main__':
  main()