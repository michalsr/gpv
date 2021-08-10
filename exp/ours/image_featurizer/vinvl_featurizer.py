import base64
import io
import json
import logging
from os.path import join
from typing import List, Tuple, Dict, Any

import cv2
import h5py
import torch
import torchvision.ops
from PIL import Image
from attr import dataclass
import torchvision.transforms as T

from exp.ours import file_paths
from exp.ours.data.gpv_data import GPVExample, Task
from exp.ours.image_featurizer.image_featurizer import ImageCollater, get_box_targets, \
  ImageFeatureExtractor, ImageRegionFeatures, numpy_xyxy_to_cxcywh,extract_query_boxes
from exp.ours.models.gpv1_preprocessing import get_stocastic_transforms
from exp.ours.models.layers import Layer
from exp.ours.util import image_utils, our_utils, py_utils
from exp.vinvl.get_vinvl import get_vinvl
from exp.vinvl import transforms as vinvl_transforms
from exp.vinvl.structures.bounding_box import BoxList
from exp.vinvl.structures.image_list import to_image_list
from torchvision.transforms import functional as F
from torch.nn import functional as nn_F
import numpy as np


class VinVLTSVReader:
  """Knows how to reading VinVL's precomputed feature TSV format"""

  def __init__(self, src):
    logging.info("Computing vinvl image-id-to-offsets")
    feature_lineidx = []
    with open(join(src, "features.lineidx")) as f:
      for line in f:
        feature_lineidx.append(int(line))

    pred_lineidx = []
    with open(join(src, "predictions.lineidx")) as f:
      for line in f:
        pred_lineidx.append(int(line))

    with open(join(src, "imageid2idx.json")) as f:
      image_id_to_idx = json.load(f)

    image_to_offset = {}
    for image_id, idx in image_id_to_idx.items():
      image_to_offset[int(image_id)] = feature_lineidx[idx], pred_lineidx[idx]
    self.image_to_offset = image_to_offset
    self.feature_file = join(src, "features.tsv")
    self.prediction_file = join(src, "predictions.tsv")

  def get(self, image_ids, return_features=True) -> List[Dict[str, Any]]:
    preds = []
    with open(self.feature_file, "r+b") as feature_f, open(self.prediction_file, "r") as pred_f:
      for image_id in image_ids:
        feature_off, pred_off = self.image_to_offset[image_id]

        pred_f.seek(pred_off)
        pred_image_id, pred = pred_f.readline().split("\t")
        assert pred_image_id == str(image_id)
        pred = json.loads(pred)

        if return_features:
          feature_f.seek(feature_off)
          parts = feature_f.readline().split(b"\t")
          assert str(image_id) == str(int(parts[0]))
          n_boxes = int(parts[1])
          ex_features = np.frombuffer(
            base64.decodebytes(parts[2]),
            dtype=np.float32).reshape((n_boxes, -1))
          # Copy to avoid trigger in annoying warning when using `torch.as_tensor` on a read-only
          # numpy array
          pred["features"] = ex_features.copy()

        preds.append(pred)
      return preds


@dataclass
class VinVLPrecomputedFeaturesCollate(ImageCollater):
  """Loads VinVL features and convert them to `ImageRegionFeatures`"""
  reader: VinVLTSVReader

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
    boxes = []
    image_sizes = []
    conf = []

    if any(x.crop is not None for x in batch) or any(x.query_box is not None for x in batch):
      raise NotImplementedError()

    preds = self.reader.get([x.image_id for x in batch])

    for pred in preds:
      w, h = pred["image_w"], pred["image_h"]
      image_sizes.append((h, w))
      image_bboxes = np.array([x["rect"] for x in pred["objects"]], dtype=np.float32)
      image_conf = torch.log(torch.as_tensor([x["conf"] for x in pred["objects"]], dtype=torch.float32))
      conf.append(image_conf)
      image_bboxes = numpy_xyxy_to_cxcywh(image_bboxes, w, h)
      boxes.append(torch.as_tensor(image_bboxes))

    n_boxes = torch.as_tensor([len(x) for x in boxes])
    box_len = n_boxes.max()

    fe = ImageRegionFeatures(
      our_utils.stack_and_pad(boxes, box_len),
      our_utils.stack_and_pad([x["features"] for x in preds], box_len),
      # -1000 so the padding is a valid value in log-probability space
      our_utils.stack_and_pad(conf, box_len, pad=-1000),
      n_boxes=n_boxes
    )
    box_targets = get_box_targets(batch, image_sizes, "cxcywh")
    return dict(features=fe), box_targets


@ImageFeatureExtractor.register("vinvl-precomputed")
@ImageFeatureExtractor.register("vinvl")
class VinVLPrecomputedFeatures(ImageFeatureExtractor):
  """Returns pre-computed VinVL features"""

  def __init__(self, model="release", dataset="coco2014trainval"):
    super().__init__()
    self._collater = None
    self.model = model
    self.dataset = dataset

  def get_collate(self, is_train=False) -> 'ImageCollater':
    if self._collater is None:
      src = join(file_paths.VINVL_SOURCE, self.model, self.dataset)
      self._collater = VinVLTSVReader(src)
    return VinVLPrecomputedFeaturesCollate(self._collater)

  def forward(self, features) -> ImageRegionFeatures:
    return features


@dataclass
class VinvlCollate(ImageCollater):
  """Collates images in way that can be passed into a VinVL model"""

  train_transforms: Dict[Task, Any]
  eval_transform: Any
  is_train: bool
  box_src: str = None
  read_image_mode: str = "vinvl"

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
    image_tensors = []
    image_sizes = []
    transforms = []
    for example in batch:
      if self.is_train:
        trans = self.train_transforms[example.task]
      else:
        trans = self.eval_transform
      transforms.append(trans)

      # Different image reading methods return subtly different outputs
      # for the same image, so we mimic the one used in vimvl (cv2.imread) by default
      if self.read_image_mode == "load_image_data":
        img, size = image_utils.load_image_data(example, None)
        img = F.to_pil_image(img)
      elif self.read_image_mode == "pil":
        img = image_utils.load_image_pil(example.image_id, example.crop)
        size = img.size[::-1]
      elif self.read_image_mode == "vinvl":
        image_f = image_utils.get_image_file(example.image_id)
        try:
          # VinVL encodes and then decodes the image due to its pre-processing setup, I have seen
          # the encoding/decoding procedure slightly alter the image (maybe due to the jpg encoding?)
          # so we do it here to 100% match with image format VinVL is trained on
          tmp = cv2.imread(image_f)
          img = Image.open(io.BytesIO(cv2.imencode('.jpg', tmp)[1])).convert('RGB')
          img = image_utils.crop_img(img, example.crop)
          size = img.size
        except cv2.error:
          # This load method fails for some formats (i.e., GIFs) due to limited support
          # of cv2.imread, we fall back to the more general load_image_data
          img, size = image_utils.load_image_data(example, None)
          img = F.to_pil_image(img).convert("RGB")
      else:
        raise NotImplementedError()
      image_sizes.append(size)
      image_tensors.append(img)

    if self.box_src:
      # Load boxes from a hdf5 source
      targets = []
      objectness = []
      with h5py.File(self.box_src, "r") as f:
        for ex, sz in zip(batch, image_sizes):
          grp = f[image_utils.get_cropped_img_key(ex.image_id, ex.crop)]
          boxes = grp["bboxes"][:]
          image_objectness = grp["objectness"][:]

          if ex.query_boxes is not None:
            qboxes, qobjectness = extract_query_boxes(
              grp, ex.image_id, ex.query_boxes, "xyxy", sz, False)
            boxes = np.concatenate([boxes, qboxes], 0)
            image_objectness = np.concatenate([image_objectness, qobjectness], 0)

          # Convert to un-normalized format expected by VinVL, boxes are already in the
          # correct xyxy format
          boxes *= np.array([sz[0], sz[1], sz[0], sz[1]])[None, :]
          objectness.append(torch.as_tensor(image_objectness))
          targets.append(BoxList(torch.as_tensor(boxes), sz))

      out = [t(img, target) for img, target, t in zip(image_tensors, targets, transforms)]
      image_tensors, targets = py_utils.transpose_lists(out)
    else:
      targets = None
      objectness = None
      image_tensors = [t(img, None)[0] for img, t in zip(image_tensors, transforms)]

    if self.box_src is None:
      # Query boxes are recorded as seperate features so clients can extract
      # features from them
      query_boxes = []
      for image_tensor, example, sz in zip(image_tensors, batch, image_sizes):
        if example.query_boxes is None:
          query_boxes.append(None)
        else:
          # Resize to be proportional to the transformed image's tenspr size in xyxy format,
          # which is the expected format for VinVL's boxes
          tensor_h, tensor_w = image_tensor.shape[-2:]
          if np.all(example.query_boxes <= 1.0):
            img_w, img_h = 1.0, 1.0  # Boxes are normalized
          else:
            img_w, img_h = sz  # Boxes are relative to original image size
          size_f = np.array([tensor_w/img_w, tensor_h/img_h, tensor_w/img_w, tensor_h/img_h])[None, :]
          boxes = torch.as_tensor(example.query_boxes * size_f, dtype=torch.float32)
          query_boxes.append(torchvision.ops.box_convert(boxes, "xywh", "xyxy"))
    else:
      # Query boxes are automatically appended to the other boxes
      query_boxes = None

    images = to_image_list(image_tensors)
    out = dict(images=images, targets=targets, objectness=objectness, query_boxes=query_boxes)
    box_targets = get_box_targets(batch, [x[::-1] for x in image_sizes])
    return out, box_targets


@ImageFeatureExtractor.register("vinvl-backbone")
class VinvlBackboneImageFeaturizer(ImageFeatureExtractor):
  """Builds by features by running a VinVL backbone on pre-computed boxes
  """

  def __init__(self, box_src, model="release", box_embed: Layer=None, freeze=None,
               train_transform=None):
    super().__init__()
    self.model = model
    self.box_embed = box_embed
    self.box_src = box_src
    self.freeze = freeze
    self.train_transform = train_transform

    vinvl, eval_transform = get_vinvl(model)
    if train_transform is None:
      train_transforms = {t: eval_transform for t in Task}
    elif train_transform == "gpv1":

      train_transforms = {}
      for task in Task:
        tr = get_stocastic_transforms(task, cls_horizontal_flip=False)
        tr = [
          vinvl_transforms.TransformWrapper(T.Compose(tr)),
          vinvl_transforms.Resize(600, 1000),
          vinvl_transforms.RandomHorizontalFlip(0.5),
          vinvl_transforms.ToTensor(),
          vinvl_transforms.Normalize(
            mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_bgr255=True),
        ]
        train_transforms[task] = vinvl_transforms.Compose(tr)
    else:
      raise NotImplementedError(train_transform)

    self.eval_transform = eval_transform
    self.train_transforms = train_transforms

    self.backbone = vinvl.backbone
    self.feature_extractor = vinvl.roi_heads["box"].feature_extractor
    self.pool = vinvl.roi_heads['box'].post_processor.avgpool
    if self.freeze == "all":
      for m in [self.backbone, self.pool, self.feature_extractor]:
        for p in m.parameters():
          p.requires_grad = False
    elif self.freeze is not None and self.freeze != "none":
      raise ValueError()

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return VinvlCollate(self.train_transforms, self.eval_transform, is_train,
                        image_utils.get_hdf5_image_file(self.box_src))

  def forward(self, images, targets, objectness, query_boxes) -> ImageRegionFeatures:
    images = to_image_list(images)
    features = self.backbone(images.tensors)
    device = images.tensors.device

    if query_boxes is not None:
      raise ValueError()

    features = self.feature_extractor(features, targets)
    features = self.pool(features)
    features = features.squeeze(-1).squeeze(-1)
    all_features = torch.split(features, [x.bbox.size(0) for x in targets])

    all_boxes = []
    for bbox, sz in zip(targets, images.image_sizes):
      w, h = bbox.size
      boxes = bbox.bbox
      boxes = boxes/torch.as_tensor([w, h, w, h], dtype=bbox.bbox.dtype, device=device).unsqueeze(0)
      # Clip since floating point issues can cause boxes to slightly exceed 1
      boxes[:, 2] = torch.clip(boxes[:, 2], 0, 1)
      boxes[:, 3] = torch.clip(boxes[:, 3], 0, 1)
      boxes = torchvision.ops.box_convert(boxes, bbox.mode, "cxcywh")
      all_boxes.append(boxes)

    out = ImageRegionFeatures.build_from_lists(all_boxes, all_features, objectness)
    if self.box_embed is not None:
      box_embed = self.box_embed(out.boxes)
      out.features = torch.cat([out.features, box_embed], -1)
    return out


class VinvlImageFeaturizer(ImageFeatureExtractor):
  """Builds by features by running a VinVL model end-to-end

  Note I am currently not sure if a loss can backprop through its outputs effectively or not
  """

  def __init__(self, model="release"):
    super().__init__()
    self.model = model
    self.vinvl, eval_transform = get_vinvl(model)
    eval_transform = eval_transform
    self.eval_transform = eval_transform
    self.train_transforms = {t: eval_transform for t in Task}

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return VinvlCollate(self.train_transforms, self.eval_transform, is_train)

  def forward(self, images, targets, query_boxes=None) -> ImageRegionFeatures:
    device = images.tensors.device

    out, backbone_features = self.vinvl(images, targets, True)

    if query_boxes is not None and any(x is not None for x in query_boxes):
      # Build BoxLists for the extra boxes we want features for
      extra_boxes = []
      for batch_ix in range(len(images.tensors)):
        extra_boxes.append(BoxList(query_boxes[batch_ix], images.image_sizes[batch_ix]))

      # Run those through the feature extractor/classification pipeline manually
      # We don't try to get these in the main `self.vinvl` call to ensure getting
      # this doesn't mess with box selection in the main method
      box_head = self.vinvl.roi_heads['box']
      query_features = box_head.feature_extractor(backbone_features, extra_boxes)
      extra_class_logits, _ = box_head.predictor(query_features)
      extra_class_logprobs = torch.log_softmax(extra_class_logits, -1)
      query_features = box_head.post_processor.avgpool(query_features).squeeze(-1).squeeze(-1)
      n_query_boxes = [len(x.bbox) for x in extra_boxes]
      query_objectness = torch.split(torch.max(extra_class_logprobs[:, 1:], -1)[0], n_query_boxes)
      query_features = torch.split(query_features, n_query_boxes)
    else:
      extra_boxes = None
      query_objectness = None
      query_features = None

    all_boxes = []
    all_features = []
    conf = []
    for batch_ix, bbox in enumerate(out):
      boxes = bbox.bbox
      features = bbox.get_field("box_features")
      scores = torch.log(bbox.get_field("scores"))
      w, h = bbox.size
      scale = torch.as_tensor([w, h, w, h], dtype=bbox.bbox.dtype, device=device).unsqueeze(0)
      boxes = boxes/scale

      if query_features is not None:
        # Append the query boxes/features/scores
        q_bbox = extra_boxes[batch_ix].bbox / scale

        # Some of the boxes can exceed 1 due to float point issues, so we clip them here
        q_bbox[:, 2] = torch.clip(q_bbox[:, 2], 0, 1.0)
        q_bbox[:, 3] = torch.clip(q_bbox[:, 3], 0, 1.0)
        boxes = torch.cat([boxes, q_bbox], 0)
        features = torch.cat([features, query_features[batch_ix]], 0)
        scores = torch.cat([scores, query_objectness[batch_ix]], 0)

      boxes = torchvision.ops.box_convert(boxes, bbox.mode, "cxcywh")
      all_boxes.append(boxes)
      all_features.append(features)
      conf.append(scores)

    return ImageRegionFeatures.build_from_lists(all_boxes, all_features, conf)

