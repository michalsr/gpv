import collections
from logging import ERROR
from typing import List, Optional, Dict
import pdb 
import numpy as np
import torch
import torchvision.ops
from allennlp.common import FromParams, Registrable, Params
from dataclasses import dataclass, asdict
from torch import nn

from exp.ours.data.dataset import Task
from exp.ours.util import our_utils
from utils.matcher import HungarianMatcher
from utils.set_criterion import SetCriterion
from torch.nn import functional as F
import os 
import utils.io as io

@dataclass
class GpvBatchLabels:
  """Labels for a batch of training data"""
  # TODO maybe we should just store GpvExample's instead

  tasks: List[Task]
  text_labels: torch.Tensor
  box_targets: List
  mil_labels: List
  segmentation_labels: List
  index_of_class: str

  # support `to` and `pin_memory` so this object can be used in a torch.DataLoader
  def to(self, device):
    return GpvBatchLabels(**{k: our_utils.to_device(v, device)
                             for k, v in asdict(self).items()})

  def pin_memory(self):
    return GpvBatchLabels(**{k: our_utils.pin_memory_recursive(v)
                             for k, v  in asdict(self).items()})


@dataclass
class GpvBatchPrediction:
  """Predictions for a batch of training data"""

  logits: torch.Tensor
  pred_boxes: torch.Tensor
  pred_rel: torch.Tensor
  n_boxes: torch.Tensor
  # TODO add the output for segmentation


class GPVLoss(Registrable, nn.Module):

  def forward(self, prediction: GpvBatchPrediction, labels: GpvBatchLabels):
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
    #to_return = ['loss_ce']
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


# class ImageContrastLoss(GPVLoss):
#   def __init__(self):
#         super().__init__()
#   def forward(self,logits,batch_labels):
#     top_1_logits = []
#     for pred in logits:
#       max_logit = torch.max(pred)
#       top_1_logits.append(max_logit)
#     final_logits = torch.FloatTensor(top_1_logits)
#     target_index = batch_labels[0]
#     one_hot = self.make_one_hot(target_index)
#     loss = F.cross_entropy(final_logits,one_hot)
#     return loss 
#   def make_one_hot(self,class_id):
#     label = torch.zeros(16)
#     label[class_id] = 1
#     return label 


@GPVLoss.register("basic-gpv-loss")
class BasicGPVLoss(GPVLoss):

  @classmethod
  def from_params(
    cls,
    params: Params,
    **kwargs
  ):
    old_weight_params = ["cap_w", "vqa_w", "cls_w", "ident_w", "webqa_w"]
    for name in old_weight_params:
      if name in params:
        assert params.pop(name) == 1.0

    loc = params["localization"]
    if "type" not in loc:
      loc["type"] = "detr-loss"
      for k in ["class_w", "bbox_w", "giou_w"]:
        loc[k] = params.pop("loc_" + k)

    return super().from_params(params, **kwargs)

  def __init__(
      self,
      localization: LocalizationLoss, sum_seq_tokens=False,
      task_weights: Optional[Dict[Task, float]] = None
  ):
    super().__init__()
    self.localization = localization
    self.sum_seq_tokens = sum_seq_tokens
    self.task_weights = task_weights
    #self.contrast_loss = contrast_loss 
  def make_one_hot(self,class_id):
    label = torch.zeros(16)
    label[class_id] = 1
    return label 
  def check_max_values(self,pred):
    max_val = -1 
    for p in pred:
      if p[1]>max_val:
        max_val = p[1]
    return max_val 
  def check_min_values(self,pred):
    min_val = 1000000000
    for p in pred:
      if p[0] < min_val:
        min_val = p[0]
    return min_val 
  def image_contrastive_loss(self,logits,index_of_class):

    final_logits = torch.cuda.FloatTensor(torch.max(logits[:,:,0],1)[0]).to("cuda:0")
    target_index = int(index_of_class)
    # for idx in batch_labels.index_of_class:
    #   if int(idx) != target_index:
    #     raise ERROR
    if not os.path.exists('exp/ours/target_index.json'):
      io.dump_json_object(target_index,'exp/ours/target_index.json')
    #losses =torch.zeros(16)
    targets = torch.zeros(final_logits.size()[0])
    targets[target_index] = 1
    targets.to("cuda:0")
    loss = F.binary_cross_entropy_with_logits(final_logits.cuda(),targets.cuda()).cuda()
    #loss = F.nll_loss(filtered_logits.cuda(),targets.type(torch.LongTensor).cuda()).cuda()
    return loss
  def text_contrastive_loss(self,logits,index_of_class):
    print(logits.size())
    final_logits = torch.cuda.FloatTensor(torch.max(logits[:,:,0],1)[0]).to("cuda:0")
    target_index = int(index_of_class)
    # for idx in batch_labels.index_of_class:
    #   if int(idx) != target_index:
    #     raise ERROR
    if not os.path.exists('exp/ours/target_index.json'):
      io.dump_json_object(target_index,'exp/ours/target_index.json')
    #losses =torch.zeros(16)
    targets = torch.zeros(final_logits.size()[0])
    targets[target_index] = 1
    targets.to("cuda:0")
    loss = F.binary_cross_entropy_with_logits(final_logits.cuda(),targets.cuda()).cuda()
    #loss = F.nll_loss(filtered_logits.cuda(),targets.type(torch.LongTensor).cuda()).cuda()
    return loss
  def mil(self,logits,batch_labels):
      # print(logits,'logits')
      # print(logits.size())
      # print(batch_labels,'batch labels')
      # print(logits[:,:,0].size())
      # print(logits[:,:,0].sort(),'sort')
      max_logits,idx = torch.max(logits[:,:,0],dim=(1))
      #print(max_logits,'max logits')
      #print(idx,'idx')
      final_logits = torch.cuda.FloatTensor(max_logits).to("cuda:0")
      #final_logits = torch.cuda.FloatTensor(torch.max(logits[:,:,0],1)[0]).to("cuda:0")
      #print(final_logits,'logits')
      loss = F.binary_cross_entropy_with_logits(final_logits.cuda(),torch.FloatTensor(batch_labels).cuda()).cuda()
      #print(loss,'loss')
      return loss 
  def syn_loss(self,logits_1,logits_2):
      #Size is Bxnum_boxesxrelevance scores
      
      logits_1 = logits_1[:,:,0]
      logits_2 = logits_2[:,:,0]
      #print(logits_2.size(),'logits')
      max_logits,ind = torch.max(logits_2,dim=(1))
      values = torch.zeros(logits_2.size())
      #print(values.size())
      #better way to do this 
      for i in range(values.size()[0]):
        values[i,ind[i]] = 1
      # values[0,ind[0]] = 1
      # values[1,ind[1]] = 1
      values = torch.FloatTensor(values).to("cuda:0")
      #print(logits_1.device)
      loss = F.binary_cross_entropy_with_logits(logits_1,values)
      return loss 

      # print(values,'values')

      # print(x,'logits')


  def forward(self, prediction: GpvBatchPrediction, batch_labels: GpvBatchLabels):
    task_to_ix = collections.defaultdict(list)
    for i, task in enumerate(batch_labels.tasks):
      task_to_ix[task].append(i)
    print(task_to_ix,'task')
    labels = batch_labels.text_labels

    losses = {}
    total_loss = 0
    
    for task, ix_lst in task_to_ix.items() :
      #print(task==Task.TEXTCONTRAST,'task')
      
      #pdb.set_trace()
      ixs = torch.as_tensor(ix_lst, device=labels.device, dtype=torch.long)
      n_boxes = prediction.n_boxes
      if task == Task.DETECTION:
        total, log = self.localization(
          prediction.pred_boxes[ixs], prediction.pred_rel[ixs],
          None if n_boxes is None else n_boxes[ixs],
          [batch_labels.box_targets[i] for i in ix_lst])
        losses.update(log)
        task_loss = total
      elif task == Task.SYNONYM:
        #print(prediction.pred_rel.size(),'size')
        ix_1 = torch.as_tensor([j for j in ix_lst if j%2 ==0],device=labels.device,dtype=torch.long)
        ix_2 = torch.as_tensor([j for j in ix_lst if j%2 ==1],device=labels.device,dtype=torch.long)
        loss = self.syn_loss(prediction.pred_rel[ix_1],prediction.pred_rel[ix_2])
        losses[str(task) + "-loss"] = loss
        task_loss = loss
      elif task == Task.MIL:
        loss = self.mil(prediction.pred_rel[ixs],batch_labels.mil_labels)
        losses[str(task) + "-loss"] = loss 
        task_loss = loss
      elif task == Task.IMAGECONTRAST:
        total_loss = torch.zeros(1).cuda()
        diff_indicies = {}
        for ind in ixs:
          if batch_labels.index_of_class[ind] != None:
            if batch_labels.index_of_class[ind] not in diff_indicies:
              diff_indicies[batch_labels.index_of_class[ind]] = []
            diff_indicies[batch_labels.index_of_class[ind]].append(ind.item())
        #print(prediction.pred_rel.size(),diff_indicies)
        for k in diff_indicies:
          loss = self.image_contrastive_loss(prediction.pred_rel[diff_indicies[k],:,:],k)
          total_loss = torch.add(total_loss,loss)
        #print(prediction.pred_rel[ixs],'relative predition')\
        # get_labels = []
        # for l in batch_labels.index_of_class:
        #   if l != None:
        #     get_labels.append(l)
        # #print('using this')
        # print(ixs,'ixs')
        # print(prediction.pred_rel[ixs].size())
        # print(batch_labels.index_of_class,'index of class')
        # loss = self.image_contrastive_loss(prediction.pred_rel[ixs],get_labels)
        losses[str(task) + "-loss"] = total_loss
        task_loss = total_loss
        #print(task_loss,'task loss')
        # total, log = self.localization(
        #   prediction.pred_boxes[ixs], prediction.pred_rel[ixs],
        #   None if n_boxes is None else n_boxes[ixs],
        #   [batch_labels.box_targets[i] for i in ix_lst])
        # losses.update(log)
        # task_loss = total
      elif task == Task.TEXTCONTRAST:
        
        total_loss = torch.zeros(1).cuda()
        diff_indicies = {}
        for ind in ixs:
          if batch_labels.index_of_class[ind] != None:
            if batch_labels.index_of_class[ind] not in diff_indicies:
              diff_indicies[batch_labels.index_of_class[ind]] = []
            diff_indicies[batch_labels.index_of_class[ind]].append(ind.item())
        print(prediction.pred_rel.size(),diff_indicies)
        for k in diff_indicies:
          loss = self.image_contrastive_loss(prediction.pred_rel[diff_indicies[k],:,:],k)
          total_loss = torch.add(total_loss,loss)
        #print(prediction.pred_rel[ixs],'relative predition')\
        # get_labels = []
        # for l in batch_labels.index_of_class:
        #   if l != None:
        #     get_labels.append(l)
        # #print('using this')
        # print(ixs,'ixs')
        # print(prediction.pred_rel[ixs].size())
        # print(batch_labels.index_of_class,'index of class')
        # loss = self.image_contrastive_loss(prediction.pred_rel[ixs],get_labels)
        losses[str(task) + "-loss"] = total_loss
        task_loss = total_loss
      elif task == Task.SEGMENTATION:
        segmentation_labels = [batch_labels.segmentation_labels[i] for i in ix_lst]
        raise NotImplementedError()
      else:
        task_labels = labels[ixs]
        task_logits = prediction.logits[ixs]
        #print(task_logits.size(),'task logit size')
        if self.sum_seq_tokens:
          task_loss = F.cross_entropy(
            task_logits.view(-1, task_logits.size(-1)), task_labels.view(-1), reduction='sum')
          task_loss = task_loss / task_logits.size(0)
        else:
          task_loss = F.cross_entropy(task_logits.view(-1, task_logits.size(-1)), task_labels.view(-1))
        losses[str(task) + "-loss"] = task_loss

      w = 1 if self.task_weights is None else self.task_weights.get(task, 1.0)
      total_loss += task_loss * w

    return total_loss, losses
