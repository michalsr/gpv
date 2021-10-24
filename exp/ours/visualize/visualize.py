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
from exp.ours.data.dataset import Task, CaptioningExample, ClsExample, LocalizationExample
from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.opensce import OpenSceDataset
from exp.ours.image_featurizer.image_featurizer import Hdf5FeatureExtractor, ImageRegionFeatures, \
  MultiHdf5FeatureExtractorCollate
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
    "border-width": border_width,
    "box-sizing": "border-box"
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


class HtmlVisualizer:

    def __init__(self, image_root, opensce_root):
        self.image_root = image_root
        self.opensce_root = opensce_root
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
        if isinstance(image_id, str) and "/" in image_id:
          src = self.opensce_root + "/" + image_id
        else:
          src = self.image_root + "/" + image_file.split("images/")[-1]
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


def get_table_html(rows):
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


def save_html(html, name, sliders=True):
  out = join(file_paths.VISUALIZATION_DIR, f"{name}.html")
  logging.info(f"Writing to {out}")
  if sliders:
    with open(join(dirname(__file__), "with_slider_template.html")) as f:
      template = string.Template(f.read())
    html = template.substitute(html_contents="\n".join(html))
  else:
    html = "\n".join(html)
  with open(out, "w") as f:
    f.write(html)
