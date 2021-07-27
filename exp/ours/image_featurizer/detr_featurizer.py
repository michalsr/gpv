from typing import Any, Dict, Tuple

import torch
import torchvision.ops
from dataclasses import dataclass

from exp.ours.data.gpv_data import Task
from exp.ours.image_featurizer.image_featurizer import ImageFeatureExtractor, \
  ImageRegionFeatures, ImageCollater, get_box_targets
from exp.ours.models.gpv1_preprocessing import get_eval_transform, get_train_transforms
from exp.ours.util import image_utils
from exp.ours.util.our_utils import get_detr_model
from utils.detr_misc import nested_tensor_from_tensor_list


@dataclass
class DetrImageCollate(ImageCollater):
  train_transforms: Dict[str, Any]
  eval_transform: Dict[str, Any]
  is_train: bool
  image_size: Tuple[int, int]
  box_format: str
  debug: bool = False

  def collate(self, examples):
    image_tensors = []
    image_sizes = []

    for example in examples:
      if self.is_train:
        trans = self.train_transforms[example.task]
      else:
        trans = self.eval_transform
      img, size = image_utils.load_image_data(example, self.image_size)
      image_tensors.append(trans(img))

      image_sizes.append(size)

    images = nested_tensor_from_tensor_list(image_tensors)

    box_targets = get_box_targets(examples, image_sizes, self.box_format)
    return dict(images=images), box_targets


@ImageFeatureExtractor.register("detr")
class PretrainedDetrFeaturizer(ImageFeatureExtractor):
  """Pretrained DETR model, used in GPV1"""

  def __init__(self, freeze_backbone=True, freeze_extractor=True, init_relevance=False,
               pretrained_model="coco_sce", clip_boxes=False, full_classifier=False):
    super().__init__()
    self.full_classifier = full_classifier
    self.init_relevance = init_relevance
    self.freeze_backbone = freeze_backbone
    self.freeze_extractor = freeze_extractor
    self.pretrained_model = pretrained_model
    self.clip_boxes = clip_boxes
    if init_relevance:
      raise NotImplementedError()
    self.detr = get_detr_model(
      pretrained=self.pretrained_model, load_object_classifier=self.full_classifier)

    self.train_transforms = {x: get_train_transforms(x) for x in Task}
    self.eval_transform = get_eval_transform()
    self.image_size = (480, 640)
    self.box_format = 'cxcywh'
    self._freeze()

  def _load_from_state_dict(self, *args, **kwargs):
    if self.detr is None:
      self.detr = get_detr_model(
        pretrained=False, load_object_classifier=self.full_classifier)
    super()._load_from_state_dict(*args, **kwargs)
    self._freeze()

  def set_freeze(self, freeze_backbone, freeze_extractor):
    self.freeze_backbone = freeze_backbone
    self.freeze_extractor = freeze_extractor
    self._freeze()

  def _freeze(self):
    for n, p in self.detr.named_parameters():
      if n.startswith("class_embed."):
        p.requires_grad = True
      if n.startswith("backbone."):
        p.requires_grad = not self.freeze_backbone
      else:
        p.requires_grad = not self.freeze_extractor

  def get_collate(self, is_train=False):
    return DetrImageCollate(
      self.train_transforms, self.eval_transform, is_train,
      self.image_size, self.box_format)

  def forward(self, images) -> ImageRegionFeatures:
    out = self.detr(images)

    boxes = out["pred_boxes"]
    if self.clip_boxes:
      # Detr can give us out-of-bound boxes, it is built so cx, cy, w, h are
      # between 0 and 1, but that can lead to invalid x1 y1 x2 y2 coordinate0s
      c = torchvision.ops.box_convert(boxes.view(-1, 4), "cxcywh", "xyxy")
      c = torch.clip(c, 0.0, 1.0)
      boxes = torchvision.ops.box_convert(c, "xyxy", "cxcywh").view(*boxes.size())

    return ImageRegionFeatures(
      boxes,
      out["detr_hs"].squeeze(0),
      out["pred_relevance_logits"]
    )

  def get_pretrained_parameters(self):
    return list(self.parameters())
