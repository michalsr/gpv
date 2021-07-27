import string
from typing import List, Dict, Optional, Union

import logging
from os.path import join, isdir, relpath, dirname, exists

import PIL.Image
import h5py
import imagesize
import numpy as np
import torch
import torchvision.ops
from PIL import Image
from dataclasses import dataclass
from torchvision.ops import box_convert

from data.coco.synonyms import SYNONYMS
from exp.ours import file_paths
from exp.ours.data.dataset import GpvDataset
from exp.ours.data.gpv_data import Task, GPVExample, GPV1_TASKS
from exp.ours.data.source_data import CocoCaptions, CocoBoxClsExample, ID_TO_COCO_CATEGORY, \
  VqaQuestion, CocoBBoxes
from exp.ours.image_featurizer.image_featurizer import Hdf5FeatureExtractor, ImageRegionFeatures
from exp.ours.image_featurizer.vinvl_featurizer import VinVLPrecomputedFeatures
from exp.ours.train.evaluator import CaptionEvaluator, vqa_score
from exp.ours.train.runner import load_gpv_predictions, GPVExampleOutput
from exp.ours.util import py_utils, image_utils
from utils import box_ops


def html_rect(x1, y1, x2, y2, rel=None, rank=None, color="black", border_width="medium"):
  rect_style = {
    "position": "absolute",
    "top": y1,
    "left": x1,
    "height": y2-y1,
    "width": x2-x1,
    "border-style": "solid",
    "border-color": color,
    "border-width": border_width
  }
  rect_style_str = "; ".join(f"{k}: {v}" for k, v in rect_style.items())

  text_style = {
    "position": "absolute",
    "top": y1-5,
    "left": x1+3,
    "color": color,
    "background-color": "black",
    "z-index": 9999,
    "padding-right": "5px",
    "padding-left": "5px",
  }
  text_style_str = "; ".join(f"{k}: {v}" for k, v in text_style.items())

  if rel is None and rank is None:
    container = ""
  else:
    container = f'class=box'
    if rel:
      container += f' data-rel="{rel}"'
    if rank:
      container += f' data-rank="{rank}"'

  html = [
    f'<div {container}>',
    f'  <div style="{rect_style_str}"></div>',
    ('' if rel is None else f'  <div style="{text_style_str}">{rel:0.2f}</div>') +
    "</div>"
  ]
  return html


def get_color_text(word, color, suffix=None):
  span = f'<span style="color: {color}">{word}</span>'
  if suffix is not None:
    span += suffix
  return span


def _html(tag, prod, style=""):
  return f'<{tag} style="{style}">{prod}</{tag}>'


@dataclass
class BoxesToVisualize:
  boxes: Union[np.ndarray, torch.Tensor]
  scores: Optional[np.ndarray]
  format: str
  color: str
  normalized: bool = False


class CocoVisualize:

    def __init__(self, image_root):
        self.image_root = image_root
        self._file_map = None

    def get_image_html_boxes(self, image_id, boxes: List[BoxesToVisualize],
                             width=None, height=None, crop=None):
      html = []
      html += [f'<div style="display: inline-block; position: relative;">']

      image_file = image_utils.get_image_file(image_id)
      if crop:
        cropped_file = image_utils.get_cropped_img_key(image_id, crop)
        cropped_file = join("cropped", cropped_file + ".jpg")
        cropped_full_path = join(file_paths.VISUALIZATION_DIR, cropped_file)
        if not exists(cropped_full_path):
          logging.info(f"Building cropped image {cropped_full_path}")
          img = PIL.Image.open(image_file)
          img = image_utils.crop_img(img, crop)
          img.save(cropped_full_path)
        src = cropped_file
        image_w, image_h = imagesize.get(cropped_full_path)
      else:
        # TODO This depends on the details of the filepath...
        src = "images/" + image_file.split("images/")[-1]
        image_w, image_h = image_utils.get_image_size(image_id)

      image_attr = dict(src=src)
      if width:
        image_attr["width"] = width
      if height:
        image_attr["height"] = height
      attr_str = " ".join(f"{k}={v}" for k, v in image_attr.items())
      html += [f'<img {attr_str}>']

      for box_set in boxes:
        if box_set.normalized:
          if width is None:
            w_factor = image_w
            h_factor = image_h
          else:
            w_factor = width
            h_factor = height
        elif width is not None:
          w_factor = width / image_w
          h_factor = height / image_h
        else:
          w_factor = 1
          h_factor = 1

        task_rel = box_set.scores
        task_boxes = box_set.boxes
        if task_rel is not None:
          ixs = np.argsort(-task_rel)
        else:
          ixs = np.arange(len(task_boxes))
        task_boxes = box_convert(torch.as_tensor(task_boxes), box_set.format, "xyxy").numpy()
        for rank, ix in enumerate(ixs):
          box = task_boxes[ix]
          rel = None if task_rel is None else task_rel[ix]
          x1, y1, x2, y2 = box
          html += html_rect(
            x1*w_factor, y1*h_factor, x2*w_factor, y2*h_factor,
            rel=rel, rank=rank+1,
            color=box_set.color,
            )

      html += [f'</div>']
      return html

    def get_image_html(self, image_id, boxes, task_boxes=None, task_rel=None,
                       color="rgb(200,0,200)", width=None, height=None, crop=None):
      to_show = []
      if boxes is not None:
        to_show.append(BoxesToVisualize(boxes, None, "xywh", "blue", False))
      if task_boxes is not None:
        to_show.append(BoxesToVisualize(task_boxes, task_rel, "cxcywh", color, True))
      return self.get_image_html_boxes(image_id, to_show, width, height, crop)

    def get_table_html(self, rows):
      html = []
      style = """
table td {
    border: thin solid; 
}
table th {
    border: thin solid;
}
      """
      html.append("<style>")
      html.append(style)
      html.append("</style>")

      html += ["<div>"]
      html += ['<table style="font-size:20px; margin-left: auto; margin-right: auto; border-collapse: collapse;">']

      all_keys = rows[0]
      for row in rows[1:]:
        all_keys.update(row)
      cols = list(all_keys)

      html += ['\t<tr>']
      for col in cols:
        html += [_html("th", col, "text-align:center")]
      html += ["\t</tr>"]

      for row in rows:
        html += [f'\t<tr>']
        for k in all_keys:
          html += [f'<td style="text-align:center">']
          val = [""] if k not in row else row[k]
          if isinstance(val, list):
            html += val
          else:
            html.append(str(val))
          html += ["</td>"]
        html += ["\t</tr>"]
      html += ["</table>"]
      html += ["</div>"]

      return html

    def _get_example_kvs(self, examples):
      if isinstance(examples[0], VqaQuestion):
        out = []
        for ex in examples:
          answer_str = ", ".join(f"{k}/{v}" for k, v in ex.answers.most_comon())
          out.append(dict(question=ex.quesiton, answer=answer_str))
        return out
      else:
        return [dict(category=ex.category) for ex in examples]

    def get_vqa_table(self, examples, pred):
      out = []
      for ex in examples:
        answer_str = ", ".join(f"{k}/{v}" for k, v in ex.answers.most_common())
        row = dict(id=ex.get_gpv_id(), question=ex.question, answer=answer_str)
        row["image"] = self.get_image_html(ex.image_id, None, width=640, height=480)
        predictions: GPVExampleOutput = pred[ex.get_gpv_id()]
        predicted_answer = predictions.text[0]
        row["score"] = "%.3f" % vqa_score(predicted_answer, ex.answers)
        row["prediction"] = predicted_answer
        row["conf"] = "%.4f" % predictions.text_logprobs[-1]
        out.append(row)
      return self.get_table_html(out)

    def get_header(self, answer, id):
      html = []
      html.append(
        f'<div style="font-size:26px; border-bottom:solid; padding-top:20px; text-align: center;">')
      html += [f"{answer} (ID={id})"]
      html.append("</div>")
      return html

    def captions_html(self, captions: List[str], unseen_concepts):
      html = []
      for caption in captions:
        html.append(self.caption_html(caption, unseen_concepts))
      return "\n".join(f"<div>{x}</div>" for x in html)

    def caption_html(self, caption: str, unseen_concepts):
      answer_words = [w.strip(".,;") for w in caption.lower().split()]
      html = []
      for w in answer_words:
        if w.lower() in unseen_concepts:
          html.append(f'<span style="color: blue">{w}</span>')
        else:
          html.append(w)
      return " ".join(html)

    def get_captioning_html(self, instance, caption, score, unseen_concepts, boxes, rel):
      html = []

      html += self.get_header("", instance["cap_id"])
      html += self.get_image_html(instance["image"], [], boxes, rel)

      answer_html = self.caption_html(instance["answer"], unseen_concepts)
      html += answer_html

      pred_html = self.caption_html(caption, unseen_concepts)

      html += [f'<span style="font-size:20px;">{score:0.2f} {pred_html}</span>']
      return html

    def get_multi_captioning_html(self, instance: CocoCaptions, names, captions, scores, unseen_concepts, boxes, rel):
      html = []

      html += self.get_header("", instance.image_id)
      html += self.get_image_html(instance.image_id, [], boxes, rel)

      html.append('<div style="font-size:20px;">')
      answer_html = self.captions_html([x.caption for x in instance.captions], unseen_concepts)
      html.append(answer_html)
      html.append('</div>')

      for name, cap, score in zip(names, captions, scores):
        pred_html = self.caption_html(cap, unseen_concepts)
        html += [f'<div><span style="font-size:20px;">{name:} {score:0.2f}, {pred_html}</span></div>']
      return html

    def get_vqa_html(self, instance, answer, unseen_concepts, boxes, rel):
      html = []

      gt = instance['all_answers']
      answer_html = []
      for w, c in sorted(gt.items(), key=lambda x: -x[1]):
        if w.lower() in unseen_concepts:
          answer_html.append(f'<span style="color: blue">{w}</span>/{c}')
        else:
          answer_html.append(f"{w}/{c}")

      query_html = []
      for word in instance["query"].split():
        if word.lower().strip("?,.") in unseen_concepts:
          query_html.append(get_color_text(word, "blue"))
        else:
          query_html.append(word)

      html += self.get_header(" ".join(query_html), instance["question_id"])
      html += self.get_image_html(instance["image"], [], boxes, rel)
      answer_str = " ".join(answer_html)
      html += ['<div style="font-size:20px">', f"Answers: {answer_str}", "</div>"]
      if vqa_score(answer, gt) > 0.0:
        color = "green"
      else:
        color = "red"
      pred_span = f'<span style="font-size:20px; color: {color}">{answer}</span>'
      html += [f'<div style="font-size:20px">Prediction={pred_span}</div>']
      return html

    def get_multi_detection_table_html(self, examples, predictions, scores=None):
      html = []
      style = """
table td {
    border: thin solid; 
}
table th {
    border: thin solid;
}
      """
      html.append("<style>")
      html.append(style)
      html.append("</style>")

      example_values = self._get_example_vals(examples)

      html += ["<div>"]
      html += ['<table style="font-size:20px; margin-left: auto; margin-right: auto; border-collapse: collapse;">']
      cols = ["image_id", "label"] + list(example_values[0].keys())
      html += ['\t<tr>']
      for col in cols:
        html += [_html("th", col, "text-align:center")]
      html += ["\t</tr>"]

      for example, example_vals in zip(examples, example_values):
        if scores is not None:
          html += [f'\t<tr>']
          html += [_html("td", str(example.image_id), "")]
          html += [_html("td", v, "") for v in example_vals.values()]
          for name, score in predictions.items():
            html += [_html("td", f"{score:0.3f}", "")]
          html += ["\t</tr>"]

        html += [f'\t<tr>']
        html += [_html("td", str(example.image_id), "")]
        html += [_html("td", v, "") for v in example_vals.values()]
        for name, pred in predictions.items():
          pred = pred[example.get_gpv_id()]
          html += [f'<td style="text-align:center">']
          html += self.get_image_html(
            example.image_id, getattr(example, "bboxes", None),
            pred.boxes, pred.relevance, 512, 348)
          html += ["</td>"]
        html += ["\t</tr>"]
      html += ["</table>"]
      html += ["</div>"]

      return html

    def get_cls_html(self, instance: CocoBoxClsExample, answers,
                     unseen_classes=None, seen_classes=None):
        html = []

        html += self.get_header(instance.category, instance.get_gpv_id())
        html += self.get_image_html(instance.image_id, [instance.box])
        labels = SYNONYMS[instance.category]

        if isinstance(answers, str):
          answer = answers
          if answer in labels:
            color = "green"
          elif answer in unseen_classes:
            color = "blue"
          elif answer in seen_classes:
            color = "black"
          else:
            color = "red"
          html += [f'<div>Answer=<span style="color: {color}">{answers}</span></div>']
        else:
          answer_html = []
          no_unseen = True
          for answer in answers:
            bold = False
            if answer in labels:
              color = "green"
              bold = True
              no_unseen = False
            elif seen_classes is None:
              color = "black"
            elif answer in seen_classes:
              color = "red"
            else:
              color = "blue"
              if no_unseen:
                bold = True
                no_unseen = False
            span = get_color_text(answer, color)
            if bold:
              span = f"<b>{span}</b>"
            answer_html += [span]

          html += ["<div>", ", ".join(answer_html), "</div>"]

        return html


def save_html(html, name, sliders=True):
  logging.info("Writing")
  if sliders:
    with open(join(dirname(__file__), "with_slider_template.html")) as f:
      template = string.Template(f.read())
    html = template.substitute(html_contents="\n".join(html))
  else:
    html = "\n".join(html)
  with open(join(file_paths.VISUALIZATION_DIR, f"{name}.html"), "w") as f:
    f.write(html)


def visualize_classification():
    instances = GpvDataset(Task.CLS, "test").load()

    src = "/Users/chris/Programming/gpv/models/t5-cls/detr1e-4-ep6/r0"
    pred = load_gpv_predictions(join(src, "eval", "gpvsce-cls-test-s1k--b20"))
    viz = CocoVisualize("images")
    instances = [x for x in instances if x.get_gpv_id() in pred]
    if len(instances) == 0:
      raise ValueError()

    np.random.RandomState(12341).shuffle(instances)
    html = []
    u_classes = GpvDataset.UNSEEN_GROUPS[Task.CLS]
    s_classes = {x for x in ID_TO_COCO_CATEGORY.values() if x not in u_classes}

    for instance in instances[:100]:
        key = instance.get_gpv_id()
        answer = pred[key].text
        print(answer)
        html += viz.get_cls_html(instance, answer, u_classes, s_classes)

    with open(join(file_paths.VISUALIZATION_DIR, "out.html"), "w") as f:
      f.write("\n".join(html))


def visualize_caption():
  SRC = "/Users/chris/Programming/gpv/models/t5-cap/lr1e-3-b60-wd1e-4/r0/eval"
  to_show = {
    "greedy": join(SRC, "gpvsce-cap-test--greedy"),
    "boost3": join(SRC, "gpvsce-cap-test--uboost3.0"),
    "boost5": join(SRC, "gpvsce-cap-test--uboost5.0")
  }

  logging.info("Loading data...")
  instances = GpvDataset(Task.CLS_IN_CONTEXT, "val", True).load()
  instances_map = {x.get_gpv_id(): x for x in instances}

  logging.info("Loading predictions...")
  predictions = {}
  for name, eval_dir in to_show.items():
    predictions[name] = load_gpv_predictions(eval_dir)

  viz = CocoVisualize("images")
  evaluator = CaptionEvaluator(bleu=None, per_caption=True)

  sample = set()
  for name, eval_dir in predictions.items():
    sample.update(eval_dir)
  is_unseen = set(x.get_gpv_id() for x in instances if sum(len(cap.meta["gpv1-unseen"]) > 0 for cap in x.captions) > 2)
  sample = sample.intersection(is_unseen)
  sample = sorted(sample)
  np.random.RandomState(1321).shuffle(sample)
  targets = set(sample[:400])
  predictions = {name: {k: v for k, v in pred.items() if k in targets}
                 for name, pred in predictions.items()}

  scores = {}
  for name, pred in predictions.items():
    to_eval = [x for x in instances if x.get_gpv_id() in pred]
    assert len(to_eval) == len(pred)
    logging.info(f"Evaluating {name} on {len(pred)} instances")
    example_scores = evaluator.evaluate_examples(to_eval, pred)
    scores[name] = {x.get_gpv_id(): sc for x, sc in zip(to_eval, example_scores)}

  with open(join(dirname(__file__), "with_slider_template.html")) as f:
    template = string.Template(f.read())

  unseen_concepts = GpvDataset.UNSEEN_GROUPS[Task.CAPTIONING]

  names = list(to_show)

  logging.info("Building evaluation")
  html = []
  for target in targets:
    instance = instances_map[target]
    captions = [predictions[p][target].text[0] for p in names]
    cap_scores = [scores[p][target]["cider"] for p in names]
    html += viz.get_multi_captioning_html(instance, names, captions, cap_scores, unseen_concepts, None, None)

  logging.info("Writing")
  with open(join(file_paths.VISUALIZATION_DIR, "out.html"), "w") as f:
    f.write(template.substitute(html_contents="\n".join(html)))


def visualize_det():
  src = "/Users/chris/Programming/gpv/models/det/"
  to_show = {
    "model": join(src, "detr-model/r0/eval/gpvsce-det-val--basic"),
    "boxes": join(src, "lr3e-4-vlr3e-5-0.1/r0/eval/gpvsce-det-val--basic"),
  }

  logging.info("Loading data")
  instances = GpvDataset(Task.DETECTION, "val").load()
  np.random.RandomState(1232).shuffle(instances)
  instances = instances[:100]

  targets = {x.get_gpv_id() for x in instances}
  predictions = {}
  for name, eval_dir in to_show.items():
    predictions[name] = load_gpv_predictions(eval_dir, load_boxes="hdf5", target_ids=targets)

  logging.info("Building evaluation")
  viz = CocoVisualize("images")
  html = viz.get_multi_detection_table_html(instances, predictions)

  save_html(html, "out")


def visualize_vqa():
  instances = GpvDataset(Task.VQA, "val").load()
  np.random.RandomState(1232).shuffle(instances)
  predictions = load_gpv_predictions("/Users/chris/tmp/gpvsce-vqa-val-s5k--basic")

  targets = [x for x in instances if x.get_gpv_id() in predictions][:100]
  assert len(targets) > 0

  logging.info("Building evaluation")
  viz = CocoVisualize("images")
  html = viz.get_vqa_table(targets, predictions)

  save_html(html, "out")


def visualize_boxes():
  box_extractors = {
    # "vinvl-pre": VinVLPrecomputedFeatures(),
    # "vinvl-hdf5": Hdf5FeatureExtractor("vinvl"),
    "base": Hdf5FeatureExtractor("vinvl-r50c4-4setvg-rboxes"),
    "all-image": Hdf5FeatureExtractor("vinvl-r50c4-4setvg-rboxes-all-image"),
  }
  instances = GpvDataset(Task.VQA, "test").load()

  for name, v in box_extractors.items():
    if isinstance(v, Hdf5FeatureExtractor):
      prev_len = len(instances)
      with h5py.File(image_utils.get_hdf5_image_file(v.source), "r") as f:
        instances = [x for x in instances if image_utils.get_cropped_img_key(x.image_id, x.crop) in f]
      if prev_len > len(instances):
        logging.info(f"Filtered {prev_len}->{len(instances)} due to extractor {name}")
        if len(instances) == 0:
          raise ValueError("All instances filtered")

  np.random.RandomState(12312).shuffle(instances)

  viz = CocoVisualize("images")

  rows = []
  for instance in instances[:10]:
    row = {}
    row["image_id"] = instance.image_id
    for k, v in box_extractors.items():
      fe = v.get_collate(False).collate([GPVExample(
        "", Task.CLS, instance.image_id,  None, crop=instance.crop,
        query_boxes=None if instance.query_box is None else np.array([instance.query_box]))])[0]
      with torch.no_grad():
        out: ImageRegionFeatures = v(**fe)
      image_boxes = out.boxes[0].numpy()
      if out.n_query_boxes is not None and out.n_query_boxes[0] > 0:
        objectness = out.objectness[0, :out.n_image_boxes[0]]
        query_boxes = image_boxes[out.n_image_boxes[0]:out.n_boxes[0]]
        image_boxes = image_boxes[:out.n_image_boxes[0]]
      else:
        objectness = out.objectness[0]
        query_boxes = None

      scores = np.exp(objectness.numpy())
      print(scores.shape)

      to_show = [
        BoxesToVisualize(image_boxes, scores, "cxcywh", "rgb(200,0,200)", True)
      ]
      if isinstance(instance, CocoBBoxes):
        to_show.append(BoxesToVisualize(instance.bboxes, None, "xywh", "blue", False))
      if query_boxes is not None:
        to_show.append(BoxesToVisualize(query_boxes, None, "cxcywh", "green", True))

      if hasattr(instance, "category"):
        row["cat"] = instance.category
      elif hasattr(instance, "question"):
        row["question"] = instance.question
      row[k] = viz.get_image_html_boxes(instance.image_id, to_show)
    rows.append(row)

  html = viz.get_table_html(rows)
  save_html(html, "out")


def visualize_all():
  model = "models/all/vinvl/r0"
  viz = CocoVisualize("images")
  for task in GPV1_TASKS:
    dataset = GpvDataset(task, "test")
    instances = dataset.load()

    if task == Task.CAPTIONING:
      instances = [x for x in instances if all(len(c.meta["gpv1-unseen"]) > 0 for c in x.captions)]
    else:
      instances = [x for x in instances if len(x.meta["gpv1-unseen"]) > 0]
    np.random.RandomState(12312).shuffle(instances)
    instances = instances[:200]
    # instances = [x for x in instances if x.get_gpv_id() == "coco-image-cap391006"]

    target_ids = set(x.get_gpv_id() for x in instances)
    predictions = load_gpv_predictions(join(model, "eval", dataset.get_name() + "--basic"), load_boxes=True, target_ids=target_ids)
    print(task)
    print(len(predictions))

    table = []
    for instance in instances:
      row = {}
      row["id"] = instance.get_gpv_id()
      pred = predictions[instance.get_gpv_id()]

      row["image"] = viz.get_image_html(
        instance.image_id,
        None if task != Task.DETECTION else instance.bboxes,
        task_rel=pred.relevance,
        task_boxes=pred.boxes,
        height=480,
        width=640,
        crop=instance.crop
      )

      if task in {Task.CLS, Task.DETECTION}:
        row["cateogry"] = instance.category
        if task == Task.CLS:
          row["accuracy"] = "1" if pred.text[0].lower() in SYNONYMS[instance.category] else "0"
      elif task == Task.VQA:
        row["query"] = instance.question
        row["answers"] = ", ".join(f"{k}/{v}" for k, v in instance.answers.most_common())
        row["score"] = "%.3f" % vqa_score(pred.text[0].lower(), instance.answers)
      elif task == Task.CAPTIONING:
        row["captions"] = [f"<div>{x.caption}<div>" for x in instance.captions]

      if task != Task.DETECTION:
        row["prediction"] = pred.text[0]
        # row["logpro"] = "%.3f" % pred.text_logprobs[0]

      table.append(row)

    table = viz.get_table_html(table)
    save_html(table, f"vinvl-unsen-{task.value}")


if __name__ == "__main__":
  py_utils.add_stdout_logger()
  visualize_boxes()
