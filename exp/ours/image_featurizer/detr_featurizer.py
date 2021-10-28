from typing import Any, Dict, Tuple, List

import torch
import torchvision.ops
from dataclasses import dataclass, replace

from exp.ours.data.dataset import Task
from exp.ours.data.gpv_example import GPVExample
from exp.ours.image_featurizer.image_featurizer import ImageFeatureExtractor, \
  ImageRegionFeatures, ImageCollater, gather_qboxes_and_targets, BoxTargets, ROIFeatureExtractor, \
  PrecomputedDataLoader
from exp.ours.models.gpv1_preprocessing import get_eval_transform, get_train_transforms
from exp.ours.models.layers import Layer
from exp.ours.util import image_utils
from exp.ours.util.our_utils import get_detr_model
from utils.detr_misc import nested_tensor_from_tensor_list


class Gpv1DetrLoader(ImageCollater):
  def __init__(self, is_train, box_extractor=None):
    super().__init__()
    self.is_train = is_train
    self.train_transforms = {task: get_train_transforms(task) for task in Task}
    self.eval_transform = get_eval_transform()
    self.image_size = (480, 640)
    self.box_extractor = box_extractor

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], BoxTargets]:
    image_tensors = []
    for example in batch:
      if self.is_train:
        trans = self.train_transforms[example.task]
      else:
        trans = self.eval_transform
      img, size = image_utils.load_image_data(example, self.image_size)
      image_tensors.append(trans(img))

    images = nested_tensor_from_tensor_list(image_tensors)
    if self.box_extractor is None:
      qboxes, targets = gather_qboxes_and_targets(batch)
      return dict(images=images, query_boxes=qboxes), targets
    else:
      regions, targets = self.box_extractor(batch)
      return dict(images=images, regions=regions), targets


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
      pretrained=self.pretrained_model, load_object_classifier=True)
    self._freeze()

  def _load_from_state_dict(self, *args, **kwargs):
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
    return Gpv1DetrLoader(is_train)

  def forward(self, images, query_boxes) -> ImageRegionFeatures:
    if any(x is not None for x in query_boxes):
      raise NotImplementedError("Query boxes not supported")
    out = self.detr(images)

    boxes = out["pred_boxes"]
    if self.clip_boxes:
      # Detr can give us out-of-bound boxes, it is built so cx, cy, w, h are
      # between 0 and 1, but that can still lead to invalid x1 y1 x2 y2 coordinates
      c = torchvision.ops.box_convert(boxes.view(-1, 4), "cxcywh", "xyxy")
      c = torch.clip(c, 0.0, 1.0)
      boxes = torchvision.ops.box_convert(c, "xyxy", "cxcywh").view(*boxes.size())

    return ImageRegionFeatures(
      boxes,
      out["detr_hs"].squeeze(0),
      out["pred_relevance_logits"]
    )


@ImageFeatureExtractor.register("detr-with-pretrained-boxes")
class BoxesWithDetrBackbone(ImageFeatureExtractor):
  def __init__(self, box_source, detr_model, feature_extractor: ROIFeatureExtractor):
    super().__init__()
    self.box_source = box_source
    self.feature_extractor = feature_extractor
    self.detr_model = detr_model

    self.backbone = get_detr_model(
      pretrained=self.detr_model, load_object_classifier=True).backbone
    self.box_ex = PrecomputedDataLoader(image_utils.get_hdf5_files(self.box_source))

  def get_collate(self, is_train=False):
    return Gpv1DetrLoader(is_train, self.box_ex)

  def forward(self, images, regions) -> ImageRegionFeatures:
    features, pos = self.backbone(images)
    src, mask = features[-1].decompose()
    features = self.feature_extractor.forward(src, regions.boxes)
    return replace(regions, features=features)
