import collections
from typing import List, Optional

import numpy as np
import torch
import torchvision.ops
from allennlp.common import FromParams, Registrable, Params
from dataclasses import dataclass
from torch import nn

from exp.ours.data.gpv_data import Task
from exp.ours.util import our_utils
from utils.matcher import HungarianMatcher
from utils.set_criterion import SetCriterion
from torch.nn import functional as F


class GPVLoss(Registrable, nn.Module):

  def forward(self, logits, labels, pred_boxes, pred_rel, n_boxes, box_targets, tasks):
    raise NotImplementedError()


class LocalizationLoss(nn.Module, Registrable):

  def forward(self, boxes, rel, n_boxes, targets):
    raise NotImplementedError()


@LocalizationLoss.register("box-cls-loss")
class BoxClsLoss(LocalizationLoss):
  def __init__(self, thresh, mode="cls"):
    super().__init__()
    self.thresh = thresh
    self.mode = mode

  def forward(self, boxes, rel, n_boxes, targets):
    losses = []
    for i, t in enumerate(targets):
      example_boxes = torchvision.ops.box_convert(boxes[i], "cxcywh", "xyxy")
      target_boxes = torchvision.ops.box_convert(targets[i], "cxcywh", "xyxy")
      # [boxes, targets]
      iou = torchvision.ops.box_iou(example_boxes, target_boxes) > self.thresh
      if self.mode == "cls":
        target = 1 - torch.any(iou, 1).to(dtype=torch.int64)
        example_rel = rel[i]
        if n_boxes is not None:
          target = target[:n_boxes[i]]
          example_rel = example_rel[:n_boxes[i]]
        losses.append(F.cross_entropy(example_rel, target))
      else:
        raise NotImplementedError()
    loss = torch.stack(losses, -1).mean()
    return loss, dict(box_cls=loss)


@LocalizationLoss.register("detr-loss")
class DetrLocalizationLoss(LocalizationLoss):

  def __init__(
    self,
    cost_class: float,
    cost_bbox: float,
    cost_giou: float,
    num_classes: int,
    eos_coef: float,
    class_w: float=None, bbox_w: float=None, giou_w: float=None,
    losses=('labels', 'boxes')
  ):
    super().__init__()
    self.class_w = cost_class if class_w is None else class_w
    self.bbox_w = cost_bbox if bbox_w is None else bbox_w
    self.giou_w = cost_giou if giou_w is None else giou_w
    self.cost_class = cost_class
    self.cost_bbox = cost_bbox
    self.cost_giou = cost_giou
    self.num_classes = num_classes
    self.eos_coef = eos_coef
    self.losses = losses
    self.matcher = HungarianMatcher(
      self.cost_class, self.cost_bbox, self.cost_giou
    )
    self.set_criterion = SetCriterion(
      num_classes=num_classes,
      matcher=self.matcher,
      weight_dict=None,
      eos_coef=eos_coef,
      losses=list(losses)
    )
    self.loc_weights = {
      "loss_ce": class_w,
      "loss_bbox": bbox_w,
      "loss_giou": giou_w,
    }

  def forward(self, boxes, rel, n_boxes, targets):
    if n_boxes is not None:
      # We make the masked boxes empty psuedo-boxes with a fixed, very low score
      # TODO do we need this now the criteion knows to account for n_boxes?
      for i, n in enumerate(n_boxes):
        boxes[i, n:, :2] = 0.0
        boxes[i, n:, 2:] = 0.001   # give boxes non-zero area so they don't NaN the loss if selected
        rel[i, n:, :-1] = -1000
        rel[i, n:, -1] = 1000

    outputs = dict(
      pred_relevance_logits=rel,
      pred_boxes=boxes
    )
    if n_boxes is not None:
      outputs[n_boxes] = n_boxes

    # Build the list-of-dictionary format the matcher expects
    target_dicts = []
    for target in targets:
      target_dicts.append(dict(
        boxes=target,
        labels=torch.zeros(target.size(0), device=target.device, dtype=torch.long)
      ))

    losses = self.set_criterion(outputs, target_dicts)
    to_return = ['loss_ce', 'loss_bbox', 'loss_giou']
    out = {}
    total_loss = 0
    for k in to_return:
      if k not in losses:
        continue
      v = losses[k]
      out[k] = v
      total_loss += v * self.loc_weights[k]
    return total_loss, out


@GPVLoss.register("basic-gpv-loss")
class BasicGPVLoss(GPVLoss):

  @classmethod
  def from_params(
    cls,
    params: Params,
    **kwargs
  ):
    loc = params["localization"]
    if "ident_w" not in params:
      params["ident_w"] = 1.0
    if "webqa_w" not in params:
      params["webqa_w"] = 1.0
    if "type" not in loc:
      loc["type"] = "detr-loss"
      for k in ["class_w", "bbox_w", "giou_w"]:
        loc[k] = params.pop("loc_" + k)
    return super().from_params(params, **kwargs)

  def __init__(
      self,
      cap_w: float, vqa_w: float, cls_w: float, ident_w: float, webqa_w: float,
      localization: LocalizationLoss, sum_seq_tokens
  ):
    super().__init__()
    self.vqa_w = vqa_w
    self.cap_w = cap_w
    self.cls_w = cls_w
    self.ident_w = ident_w
    self.webqa_w = webqa_w
    self.localization = localization
    self.sum_seq_tokens = sum_seq_tokens
    self.ce_weight_table = {
      Task.CAPTIONING: cap_w,
      Task.VQA: vqa_w,
      Task.CLS: cls_w,
      Task.CLS_IN_CONTEXT: ident_w,
      Task.WEBQA: webqa_w,
    }

  def forward(self, logits, labels, pred_boxes, pred_rel, n_boxes, box_targets, tasks):
    task_to_ix = collections.defaultdict(list)
    for i, task in enumerate(tasks):
      task_to_ix[task].append(i)

    losses = {}
    total_loss = 0
    for task, ix_lst in task_to_ix.items() :
      ixs = torch.as_tensor(ix_lst, device=labels.device, dtype=torch.long)
      if task == Task.DETECTION:
        total, log = self.localization(
          pred_boxes[ixs], pred_rel[ixs],
          None if n_boxes is None else n_boxes[ixs],
          [box_targets[i] for i in ix_lst])
        losses.update(log)
        total_loss += total
      else:
        task_labels = labels[ixs]
        task_logits = logits[ixs]
        if self.sum_seq_tokens:
          task_loss = F.cross_entropy(
            task_logits.view(-1, task_logits.size(-1)), task_labels.view(-1), reduction='sum')
          task_loss = task_loss / logits.size(0)
        else:
          task_loss = F.cross_entropy(task_logits.view(-1, task_logits.size(-1)), task_labels.view(-1))
        losses[str(task) + "-loss"] = task_loss
        total_loss += task_loss * self.ce_weight_table[task]

    return total_loss, losses
