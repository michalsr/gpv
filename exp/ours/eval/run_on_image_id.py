"""This script demos how to use our model on a particular image"""
import argparse
from typing import List, Dict, Any

import torch.cuda

from exp.ours.data.gpv_example import GPVExample
from exp.ours.models.model import GPVExampleOutput
from exp.ours.train.runner import BeamSearchSpec
from exp.ours.util import our_utils, py_utils
from exp.ours.util.load_model import load_model
from exp.ours.util.our_utils import select_run_dir


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model", help="Model directory to run with")
  parser.add_argument("image_id",
                      help="Image id to use, it has to be something the model can "
                           "locate in its HDF5 feature files")
  parser.add_argument("query", help="Query to give the model")
  parser.add_argument("--answer_options", nargs="+",
                      help="Answer options to use")
  args = parser.parse_args()

  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  run = select_run_dir(args.model)
  print(f"Loading model from {run}...")
  model = load_model(run, device=device)

  print("Done, setting up inputs...")
  try:
    # For coco-images the image_id is an integer
    image_id = int(args.image_id)
  except ValueError:
    image_id = args.image_id

  if args.answer_options:
    # This tell the model what the answer options are, and to do re-ranking
    # It only needs to be called once if you are running on multiple batches
    print(f"Options set to {args.answer_options}")
    model.set_prediction_args(
      answer_options=args.answer_options,
      rerank_answer_options=True
    )
  else:
    # Tell the model to do beam search
    model.set_prediction_args(
      beam_search_spec=BeamSearchSpec(beam_size=20, max_seq_len=30)
    )

  query_input = GPVExample(
    id="", task=None,
    image_id=image_id,
    query=args.query,
  )
  input_batch = [query_input]
  input_batch = [model.preprocess_example(x) for x in input_batch]
  features: Dict[str, Any] = model.get_collate()(input_batch)
  features = our_utils.to_device(features, device)

  print("Starting prediction...")
  batch_output: List[GPVExampleOutput] = model.predict(**features)

  print("Done, printing outputs")
  output = batch_output[0]
  for text, prob in zip(output.text, output.text_logprobs):
    print(f"{text}: {prob:.5g}")


if __name__ == '__main__':
  py_utils.add_stdout_logger()
  main()