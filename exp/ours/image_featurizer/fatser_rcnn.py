import logging
from typing import List

import torch
import torchvision
from detectron2 import model_zoo
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.model_zoo import get_config
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as F

from exp.ours.data.gpv_example import GPVExample
from exp.ours.image_featurizer.image_featurizer import ImageFeatureExtractor, ImageRegionFeatures, \
  ImageCollater
from exp.ours.util import image_utils, our_utils, py_utils


class LoadImagesCollater(ImageCollater):
  def __init__(self, transforms=None):
    self.transforms = transforms

  def collate(self, batch: List[GPVExample]):
    images = []
    for ex in batch:
      image_file = image_utils.get_image_file(ex.image_id)
      img = image_utils.load_image_pil(image_file, ex.crop)
      if self.transforms is not None:
        img = self.transforms(img)
      else:
        img = F.to_tensor(img)
      images.append(img)
    return dict(images=images), None


class FasterRCNNFeaturizer(ImageFeatureExtractor):

  def __init__(self, backbone="fasterrcnn_resnet50_fpn", init_from=None, num_classes=2,
               min_size=600, max_size=1000):
    super().__init__()
    self.backbone = backbone
    self.init_from = init_from
    if init_from == "torchvision":
      pretrained = False
    else:
      pretrained = True

    fpn = True
    if self.backbone == "fasterrcnn_resnet50_fpn":
      bk = resnet_fpn_backbone('resnet50', pretrained)
    elif self.backbone == "resnext101_32x8d_fpn":
      bk = resnet_fpn_backbone('resnext101_32x8d', pretrained)
    elif self.backbone == "resnext101_32x8d":
      return_layers = {'layer4': "0"}
      norm_layer = FrozenBatchNorm2d
      bk = getattr(torchvision.models, 'resnext101_32x8d')(
        replace_stride_with_dilation=[False, False, False],
        pretrained=True, norm_layer=norm_layer)
      bk = IntermediateLayerGetter(bk, return_layers=return_layers)
      bk.out_channels = 2048
      fpn = False
    else:
      raise NotImplementedError(self.backbone)

    if fpn:
      self.model = FasterRCNN(bk, num_classes, min_size=min_size, max_size=max_size)
    else:
      anchor_sizes = ((32, 64, 128, 256, 512),)
      aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
      rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
      self.model = FasterRCNN(bk, num_classes, min_size=min_size, max_size=max_size,
                              rpn_anchor_generator=rpn_anchor_generator)

    if self.init_from is not None:
      state = torch.load(self.init_from)["model"]
      for k in list(state):
        if k.startswith("roi_heads.mask_head") or k.startswith("roi_heads.mask_predictor"):
          del state[k]
      logging.info(f"Initializing model from {self.init_from}")
      self.model.load_state_dict(state)
    self.model.eval()

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return LoadImagesCollater()

  def forward(self, **kwargs) -> ImageRegionFeatures:
    model_out = self.model(**kwargs)
    boxes = []
    scores = []
    img0 = kwargs["images"][0]
    dtype, device = img0.dtype, img0.device
    for i, image_out in enumerate(model_out):
      h, w = kwargs["images"][i].size()[1:]
      div = torch.as_tensor([w, h, w, h], dtype=dtype, device=device)
      img_boxes = image_out["boxes"] / div.unsqueeze(0)
      # fasterrcnn can return boxes that are marginally out-of-bounds, fix that here
      img_boxes = torch.clip(img_boxes, min=0.0, max=1.0)
      img_boxes = torchvision.ops.box_convert(img_boxes, "xyxy", "cxcywh")
      boxes.append(img_boxes)
      scores.append(torch.log(image_out["scores"]))
    return ImageRegionFeatures.build_from_lists(boxes, None, scores)

