import argparse
import json
from os.path import join

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from data.coco.synonyms import SYNONYMS
from exp.ours.data.dataset import Task, ClsExample, VqaExample, CaptioningExample, \
  LocalizationExample, GPV2_TASKS
from exp.ours.data.gpv import COCO_CATEGORIES, GpvDataset
from exp.ours.data.opensce import OpenSceDataset, OPENSCE_SYNONYMS
from exp.ours.data.webqa import WebQaDataset, WebQaAnswers
from exp.ours.models.model import GPVExampleOutput
from exp.ours.train.evaluator import vqa_score, LocalizationEvaluator, OpenSceVqaEvaluator
from exp.ours.train.runner import load_gpv_predictions
from exp.ours.util import py_utils, image_utils
from exp.ours.util.py_utils import FindAll


def build_webqa():
  dataset = WebQaDataset("val")

  models = {
    # "vinvl": ("models/gpv2-tasks/vinvl/", "basic"),
    "vinvl-web-fifth": ("models/gpv2-tasks/vinvl-web-fifth/", "basic")
  }
  instances = dataset.load()
  np.random.shuffle(instances)

  predictions = {}
  for k, (src, eval_name) in models.items():
    predictions[k] = load_gpv_predictions(
      join(src, "r0", "eval", f"{dataset.get_name()}--{eval_name}"),
      load_boxes=False,
    )

  coco_seen = set(py_utils.flatten_list(SYNONYMS[x] for x in GpvDataset.UNSEEN_GROUPS[dataset.get_task()]))
  all_coco = set(py_utils.flatten_list(SYNONYMS[x] for x in COCO_CATEGORIES))
  data = []
  for instance in instances:
    row = {}
    row["image"] = instance.image_id
    row["category"] = instance.target_answer
    if instance.target_answer in coco_seen:
      kind = "cls-unseen"
    elif instance.target_answer in all_coco:
      kind = "cls-seen"
    else:
      kind = "unseen"
    row["kind"] = kind
    syns = OPENSCE_SYNONYMS.get(instance.target_answer, [instance.target_answer])
    for model_name, pred in predictions.items():
      output = pred[instance.get_gpv_id()]
      answers = output.text
      row[f"{model_name}"] = answers[0]
      row[f"{model_name}-conf"] = output.text_logprobs[0]
      row[f"{model_name}-accuracy"] = float(answers[0] in syns)
      for rank, answer in enumerate(answers[:5]):
        row[f"{model_name}-{rank+1}"] = answer
    data.append(row)

  with open("/var/chrisc/webqa.json", "w") as f:
    json.dump(data, f, indent=2)


def get_answer_type(answer, webqa_words):
  try:
    float(answer)
    return "num"
  except ValueError:
    pass

  if answer in {"one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten"}:
    return "num"

  if answer in {
    "red", "tan", "blue", "white", "black", "brown", "yellow",
    "orange", "green", "pink", "grey", "purple",
  }:
    return "color"

  if any(x in set(COCO_CATEGORIES) for x in answer.split()):
    return "in-coco"

  if any(x in answer for x in webqa_words):
    return "in-webqa"

  return "other"


def build_opensce(task: Task, sample=None):
  dataset = OpenSceDataset(task, "val")

  models = {
    "vinvl+web": ("models/gpv2/web4/p0.2-ep8/", "basic"),
    "vinvl+web+boxloc": ("models/gpv2/per-box/basic-0.2-ep8", "basic"),
  }
  instances = dataset.load()
  np.random.shuffle(instances)

  if sample is not None:
    instances = instances[:sample]
    target_ids = set(x.get_gpv_id() for x in instances)
  else:
    target_ids = None
  # instances = [x for x in instances if x.get_gpv_id() == "opensce-loc-2176"]
  # target_ids = [x.get_gpv_id() for x in instances]

  predictions = {}
  for k, (src, eval_name) in models.items():
    predictions[k] = load_gpv_predictions(
      join(src, "r0", "eval", f"{dataset.get_name()}--{eval_name}"),
      load_boxes=task == Task.DETECTION,
      target_ids=target_ids
    )

  loc_eval = LocalizationEvaluator()
  vqa_eval = OpenSceVqaEvaluator(spacy_lemmatizer=False)

  all_coco = set(py_utils.flatten_list(SYNONYMS[x] for x in COCO_CATEGORIES))
  webqa = set(WebQaAnswers())

  targets = {}
  for cat in COCO_CATEGORIES:
    for syn in SYNONYMS[cat]:
      targets[syn] = cat
  coco_finder = FindAll(targets)

  data = []
  for instance in tqdm(instances, ncols=100):
    row = {}
    row["id"] = instance.get_gpv_id()
    row["image_id"] = instance.image_id

    if isinstance(instance, ClsExample) and (instance.query_box is not None or instance.crop is not None):
      box_data = {}
      if instance.query_box is None:
        x1, y1, w, h = instance.crop
      else:
        x1, y1, w, h = instance.query_box
      row["box-area"] = w*h
      for name, val in zip(["x1", "y1", "x2", "y2"], [x1, y1, x1+w, y1+h]):
        box_data[name] = float(val)
      box_data["color"] = "green"

      row["image"] = dict(image=instance.image_id, boxes=[box_data])

    elif isinstance(instance, LocalizationExample):
      boxes = []
      for box in instance.bboxes:
        box_data = {}
        x1, y1, w, h = box
        for name, val in zip(["x1", "y1", "x2", "y2"], [x1, y1, x1+w, y1+h]):
          box_data[name] = float(val)
        box_data["color"] = "green"
        boxes.append(box_data)
      row["image"] = dict(image=instance.image_id, boxes=boxes)

    else:
      row["image"] = instance.image_id

    if isinstance(instance, ClsExample) or isinstance(instance, LocalizationExample):
      row["category"] = instance.category
      if instance.category in all_coco:
        kind = "in-coco"
      elif instance.category in webqa:
        kind = "in-webqa"
      else:
        kind = "in-none"
      row["kind"] = kind
    elif isinstance(instance, VqaExample):
      row["question"] = instance.question
      row["answer-type"] = get_answer_type(instance.answers, webqa)
      assert isinstance(instance.answers, str)
      row["answer"] = instance.answers
      if coco_finder.find(instance.answers):
        row["coco-answer"] = "True"
      else:
        row["coco-answer"] = "False"
    elif isinstance(instance, CaptioningExample):
      pass
    else:
      raise ValueError()

    for model_name, pred in predictions.items():
      output = pred[instance.get_gpv_id()]
      answers = output.text

      if task != Task.DETECTION:
        row[model_name] = answers[0]
        row[f"{model_name}:conf"] = output.text_logprobs[0]
        if len(answers) > 0:
          table = []
          for ans, lp in zip(answers[:5], output.text_logprobs[:5]):
            table.append(dict(ans=ans, prob=np.exp(lp)*100))
          row[f"{model_name}:answers"] = table

      else:
        metrics = loc_eval.evaluate_examples(
          [instance], {instance.get_gpv_id(): output}, return_pr=True)[0]
        row[f"{model_name}:AP"] = metrics["AP"]

        missing = torchvision.ops.box_convert(torch.as_tensor(instance.bboxes), "xywh", "xyxy")

        pred_boxes = torchvision.ops.box_convert(torch.as_tensor(output.boxes), "cxcywh", "xyxy")

        boxes = []
        for ix in np.argsort(-output.relevance)[:10]:
          box_data = {}
          x1, y1, x2, y2 = pred_boxes[ix].numpy()
          for name, val in zip(["x1", "y1", "x2", "y2"], [x1, y1, x2, y2]):
            box_data[name] = float(val)
          box_data["rel"] = float(output.relevance[ix])
          boxes.append(box_data)

          iou = torchvision.ops.box_iou(pred_boxes[ix:ix+1], missing)[0]
          missing = missing[iou < 0.5]

          if torch.any(iou > 0.5):
            box_data["color"] = "green"
          else:
            box_data["color"] = "red"
          if len(missing) == 0:
            break

        row[model_name] = dict(image=instance.image_id, boxes=boxes)

      if isinstance(instance, CaptioningExample):
          row[f"{model_name}:has-coco-concept"] = str(bool(coco_finder.find(output.text[0])))

      elif isinstance(instance, ClsExample):
        syns = OPENSCE_SYNONYMS.get(instance.category, [instance.category])
        row[f"{model_name}:accuracy"] = float(answers[0] in syns)
      elif isinstance(instance, VqaExample):
        res = vqa_eval.evaluate_examples([instance], {instance.get_gpv_id(): output})[0]
        row[f"{model_name}:acc"] = float(res["acc"])
        row[f"{model_name}:iou"] = float(res["iou"])
        if coco_finder.find(output.text[0]):
          row[f"{model_name}:coco-answer"] = "True"
        else:
          row[f"{model_name}:coco-answer"] = "False"

    data.append(row)

  with open(f"/var/chrisc/concept-lens/{task.value}.json", "w") as f:
    json.dump(data, f, indent=2)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  parser.add_argument("--sample", type=int)
  args = parser.parse_args()

  py_utils.add_stdout_logger()
  build_opensce(Task(args.task), args.sample)


if __name__ == '__main__':
  main()

