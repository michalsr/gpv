import argparse
from collections import Counter
from os.path import join

from allennlp.common import Params

from exp.ours.data.dataset import Dataset, ClsExample, Task, VqaExample
from exp.ours.train.runner import load_gpv_predictions
from exp.ours.util import py_utils, our_utils
from exp.ours.visualize.visualize import HtmlVisualizer, BoxesToVisualize, get_color_text, save_html
from utils.io import load_json_object

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("eval")
  parser.add_argument("--sample", default=200, type=int)

  py_utils.add_stdout_logger()
  args = parser.parse_args()

  eval_dir = args.eval
  config = load_json_object(join(args.eval, "config.json"))
  our_utils.import_all()
  dataset = Dataset.from_params(Params(config["dataset"]))
  examples = dataset.load()
  examples.sort(key=lambda x: x.get_gpv_id())
  np.random.RandomState(56234).shuffle(examples)
  examples = examples[:args.sample]
  predictions = load_gpv_predictions(eval_dir)

  viz = HtmlVisualizer("images", "opensce-images")
  table = []
  for ex in examples:
    pred = predictions[ex.get_gpv_id()]
    row = dict()

    boxes_to_show = []
    if isinstance(ex, ClsExample) and ex.query_box is not None:
      boxes_to_show.append(BoxesToVisualize(
        boxes=np.array([ex.query_box]), format="xywh", normalized=True, color="blue", scores=None))
    row["image"] = viz.get_image_html_boxes(ex.image_id, boxes_to_show, crop=ex.crop)

    if isinstance(ex, ClsExample):
      row["category"] = ex.category
      pred_html = ["<ul"]
      for answer in pred.text[:5]:
        if answer == ex.category:
          color = "green"
        else:
          color = None
        pred_html += [f"<li>{get_color_text(answer, color)}</li>"]
      pred_html += ["<ul"]
      row["pred"] = "\n".join(pred_html)

    elif isinstance(ex, VqaExample):
      row["question"] = ex.question
      if isinstance(ex.answers, Counter):
        raise ValueError()
      row["answer"] = ex.answers
      row["prediction"] = ex.text[0]
    else:
      raise NotImplementedError()
    table.append(row)

  table = viz.get_table_html(table)
  save_html(table, "out")


if __name__ == '__main__':
  main()