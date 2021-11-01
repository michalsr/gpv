import logging
from typing import List, Dict, Optional, Union
import numpy as np

import torch
import torchvision
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model, StandardROIHeads
from detectron2.modeling.roi_heads import Res5ROIHeads
from torchvision import transforms as T
from dataclasses import replace
from detectron2 import model_zoo
from detectron2.model_zoo import get_config
from detectron2.structures import ImageList, Instances, Boxes
from torchvision.transforms import ColorJitter, RandomGrayscale, Compose

from exp.ours.data.dataset import Task, GPV2_TASKS
from exp.ours.data.gpv_example import GPVExample
from exp.ours.image_featurizer.image_featurizer import ImageFeatureExtractor, ImageCollater, \
  ROIFeatureExtractor, PrecomputedDataLoader, gather_qboxes_and_targets, ImageRegionFeatures
from detectron2.data.transforms import ResizeShortestEdge

from exp.ours.models.gpv1_preprocessing import get_stocastic_transforms
from exp.ours.models.layers import Layer
from exp.ours.util import image_utils, py_utils


class DetectronCollater(ImageCollater):
  def __init__(
      self,
      cfg,
      box_source: PrecomputedDataLoader=None,
      is_train=False,
      transforms="gpv1"
  ):
    self.resize = ResizeShortestEdge(
      [cfg.MIN_SIZE_TEST, cfg.MIN_SIZE_TEST], cfg.MAX_SIZE_TEST
    )
    self.box_source = box_source
    self.transforms = transforms

    self._transforms = None
    if is_train:
      if self.transforms == "gpv1":
        self._transforms = {t: T.Compose([T.ToPILImage(mode='RGB')] + get_stocastic_transforms(t)) for t in GPV2_TASKS}
      elif self.transforms is None:
        pass
      else:
        raise NotImplementedError()
    self.input_format = cfg.FORMAT

  def collate(self, batch: List[GPVExample]):
    inputs = []
    for ex in batch:
      image_file = image_utils.get_image_file(ex.image_id)
      # Image in (H, W, C)
      img, _ = image_utils.load_image_ndarray(image_file, crop=ex.crop)
      if self._transforms:
        img = np.array(self._transforms[ex.task](img))

      if self.input_format == "RGB":
        img = img[:, :, ::-1]

      img = self.resize.get_transform(img).apply_image(img)
      img = torch.tensor(img.transpose(2, 0, 1))

      inputs.append(dict(image=img, width=1.0, height=1.0))

    if self.box_source is None:
      qboxes, targets = gather_qboxes_and_targets(batch)
      return dict(images=inputs, query_boxes=qboxes), targets
    else:
      regions, targets = self.box_source(batch)
      return dict(images=inputs, regions=regions), targets


class DetectronObjectDetector(ImageFeatureExtractor):

  def __init__(self, name, pretrained=True, one_class=False):
    super().__init__()
    self.name = name
    logging.info(f"Loading model {name}")
    with py_utils.DisableLogging():
      cfg = get_config(name, True)
      # cfg.device = "cpu"
      # print("DEBUG")
      if one_class:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        assert not pretrained
        self.model = build_model(cfg)
      else:
        self.model = model_zoo.get(name, pretrained)
      self.model.eval()
    self.input_cfg = cfg.INPUT

  def train(self, mode: bool = True):
    pass

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return DetectronCollater(self.input_cfg)

  def forward(self, images, query_boxes) -> ImageRegionFeatures:
    if query_boxes is not None and any(x is not None for x in query_boxes):
      raise NotImplementedError()
    model_out = self.model(images)
    all_boxes = []
    all_scores = []
    for i, image_out in enumerate(model_out):
      instances = image_out["instances"]
      boxes = instances.pred_boxes.tensor
      w, h = instances.image_size
      div = torch.as_tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
      img_boxes = boxes / div.unsqueeze(0)
      # fasterrcnn can return boxes that are marginally out-of-bounds, fix that here
      img_boxes = torch.clip(img_boxes, min=0.0, max=1.0)
      img_boxes = torchvision.ops.box_convert(img_boxes, "xyxy", "cxcywh")
      all_boxes.append(img_boxes)
      all_scores.append(torch.log(instances.scores))
    return ImageRegionFeatures.build_from_lists(all_boxes, None, all_scores)


@ImageFeatureExtractor.register("detectron-backbone")
class DetectronBackboneWithBoxes(ImageFeatureExtractor):
  def __init__(self, box_source, detectron_model,
               freeze: str="all", coordinate_embedder: Optional[Layer]=None):
    super().__init__()
    self.box_source = box_source
    self.detectron_model = detectron_model
    self.freeze = freeze
    self.coordinate_embedder = coordinate_embedder

    logging.info(f"Loading model {detectron_model}")
    with py_utils.DisableLogging():
      # Keep the import here for now so the dependency is optional
      from detectron2 import model_zoo
      # print(type(model_zoo.get(detectron_model, True)))
      model = model_zoo.get(detectron_model, True)
      self.register_buffer("pixel_mean", model.pixel_mean, False)
      self.register_buffer("pixel_std", model.pixel_std, False)
      self.backbone = model.backbone
      self.input_cfg = model_zoo.get_config(detectron_model, True).INPUT

    roi = model.roi_heads

    if isinstance(roi, StandardROIHeads):
      self.box_pooler = roi.box_pooler
      self.box_head = roi.box_head
      self.box_in_features = roi.box_in_features
      self.roi = "StandardROIHeads"
    elif isinstance(roi, Res5ROIHeads):
      self.in_features = roi.in_features
      self.pooler = roi.pooler
      self.res5 = roi.res5
      self.roi = "Res5ROIHeads"
    else:
      raise NotImplementedError()

    if self.freeze == "all":
      for param in self.backbone.parameters():
        param.requires_grad = False
    elif self.freeze == "none" or self.freeze is None:
      pass
    else:
      raise NotImplementedError(self.freeze)
    files = image_utils.get_hdf5_files(self.box_source)
    assert len(files) > 0
    self.loader = PrecomputedDataLoader(files)

  def get_collate(self, is_train=False):
    return DetectronCollater(self.input_cfg, self.loader, is_train)

  def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
    """
    Normalize, pad and batch the input images.
    """
    images = [x["image"] for x in batched_inputs]
    images = [(x - self.pixel_mean) / self.pixel_std for x in images]
    images = ImageList.from_tensors(images, self.backbone.size_divisibility)
    return images

  def forward(self, images, regions):
    images = self.preprocess_image(images)
    features = self.backbone(images.tensor)

    batch = len(regions.boxes)

    boxes = torchvision.ops.box_convert(regions.boxes.view(-1, 4), "cxcywh", "xyxy").view(batch, -1, 4)
    if regions.n_boxes is None:
      proposed_boxes = torch.unbind(boxes, 0)
    else:
      proposed_boxes = []
      for i in range(len(boxes)):
        proposed_boxes.append(boxes[i, :regions.n_boxes[i]])

    # Scale to match the image sizes
    for img_boxes, (H, W) in zip(proposed_boxes, images.image_sizes):
      div = torch.as_tensor([W, H, W, H], device=img_boxes.device, dtype=img_boxes.dtype)
      img_boxes *= div
    proposed_boxes = [Boxes(x) for x in proposed_boxes]

    if self.roi == "StandardROIHeads":
      features = [features[f] for f in self.box_in_features]
      box_features = self.box_pooler(features, proposed_boxes)
      region_features = self.box_head(box_features)
    elif self.roi == "Res5ROIHeads":
      region_features = self.pooler([features[f] for f in self.in_features], proposed_boxes)
      region_features = self.res5(region_features)
      region_features = region_features.mean([2, 3])
    else:
      raise NotImplementedError()

    # Convert to [batch, boxes_per_batch, dim]
    if regions.n_boxes is not None:
      out = torch.zeros(
        regions.boxes.size(0), regions.boxes.size(1), region_features.size(-1),
        device=region_features.device, dtype=region_features.dtype
      )
      on = 0
      for i in range(batch):
        end = on + regions.n_boxes[i]
        out[i, :regions.n_boxes[i]] = region_features[on:end]
        on = end
      assert end == region_features.size(0)
      region_features = out
    else:
      region_features = region_features.view(
        batch, regions.boxes.size(1), region_features.size(-1))

    if self.coordinate_embedder is not None:
      box_embed = self.coordinate_embedder(regions.boxes)
      region_features = torch.cat([region_features, box_embed], -1)

    return replace(regions, features=region_features)