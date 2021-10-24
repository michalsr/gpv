import logging
from contextlib import ExitStack
from os.path import join, exists
from typing import List, Dict, Any, Tuple, Optional, NewType

import logging
from typing import List, Dict, Any, Tuple, Optional

import h5py
import numpy as np
import torch
import torchvision
from allennlp.common import Registrable, Params
from dataclasses import dataclass, replace
from torch import nn
from torchvision.transforms.functional import hflip
from torch.nn import functional as F
from exp.ours import file_paths
from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.dataset import Task
from exp.ours.models.gpv1_preprocessing import get_train_transforms, get_eval_transform
from exp.ours.models.layers import Layer
from exp.ours.util import image_utils, py_utils, our_utils
from exp.ours.util.image_utils import get_hdf5_image_file
from utils.detr_misc import nested_tensor_from_tensor_list, NestedTensor


@dataclass
class ImageRegionFeatures:

  @staticmethod
  def build_from_lists(boxes, features, objectness):
    n_boxes = [len(x) for x in boxes]
    max_len = max(n_boxes)
    n_boxes = torch.as_tensor(n_boxes, dtype=torch.long, device=boxes[0].device)
    return ImageRegionFeatures(
      our_utils.stack_and_pad(boxes, max_len),
      our_utils.stack_and_pad(features, max_len),
      # -10000 so the padding is a valid log-probability
      our_utils.stack_and_pad(objectness, max_len, -10000),
      n_boxes,
    )

  boxes: torch.Tensor
  """[batch, n_regions, 4] boxes in [cx, cy, w, y] format normalized between 0 and 1"""

  features: torch.Tensor
  """[batch, n_regions, n_features] region features"""

  objectness: Optional[torch.Tensor]
  """[batch, n_regions] or [batch, n_regions, n_classes] objectness log-probability"""

  n_boxes: Optional[torch.Tensor] = None
  """[batch] number of boxes for each batch if batches can have differing numbers of boxes"""

  def numpy(self):
    return ImageRegionFeatures(
      self.boxes.cpu().numpy(),
      None if self.features is None else self.features.cpu().numpy(),
      None if self.objectness is None else self.objectness.cpu().numpy(),
      None if self.n_boxes is None else self.n_boxes.cpu().numpy()
    )

  def to(self, device):
    return ImageRegionFeatures(
      self.boxes.to(device),
      None if self.features is None else self.features.to(device),
      None if self.objectness is None else self.objectness.to(device),
      None if self.n_boxes is None else self.n_boxes.to(device)
    )


BoxTargets = NewType('BoxTargets', List[Optional[torch.Tensor]])
"""Batch of target boxes in cxcywh format, normalized between 0 and 1"""


class ImageFeatureExtractor(nn.Module, Registrable):
  """Extracts regions and region feature vectors for an image"""

  def get_collate(self, is_train=False) -> 'ImageCollater':
    raise NotImplementedError()

  def forward(self, **kwargs) -> ImageRegionFeatures:
    raise NotImplementedError()


class ImageCollater:

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], BoxTargets]:
    """
    return:
      image_inputs: Inputs to pass to `forward`
      box_targets: Boxes to use if learning detection, these are constructed here since
                   the image pre-processing might require adjusting that the target boxes
                   (e.g., horizontal flipping will require flipping the target bosxes)
    """
    raise NotImplementedError()


class ROIFeatureExtractor(Layer):
  """Extract image features for a given set of regions"""

  def forward(self, x: NestedTensor, boxes: torch.Tensor):
    raise NotImplementedError()


def get_box_targets(examples: List[GPVExample], image_sizes, box_format="cxcywh") -> BoxTargets:
  all_boxes = []
  for i, example in enumerate(examples):
    if example.target_boxes is None:
      all_boxes.append(None)
    else:
      boxes = torch.as_tensor(example.target_boxes, dtype=torch.float)
      x, y, w, h = boxes.split([1, 1, 1, 1], 1)
      if box_format == "xyxy":
        box_parts = [x, y, x + w, y + h]
      elif box_format == "cxcywh":
        box_parts = [x + 0.5*w, y + 0.5*h, w, h]
      else:
        raise NotImplementedError(box_format)
      H, W = image_sizes[i]
      boxes = torch.cat([
        box_parts[0] / W,
        box_parts[1] / H,
        box_parts[2] / W,
        box_parts[3] / H
      ], 1)
      all_boxes.append(boxes)
  return all_boxes


def numpy_xyxy_to_cxcywh(image_boxes, w, h):
  boxes_w = image_boxes[:, 2] - image_boxes[:, 0]
  boxes_h = image_boxes[:, 3] - image_boxes[:, 1]
  converted_boxes = np.stack([
    (image_boxes[:, 0] + boxes_w/2) / w,
    (image_boxes[:, 1] + boxes_h/2) / h,
    boxes_w/w,
    boxes_h/h
  ], -1)
  return converted_boxes


def numpy_xywh_to_cxcywh(image_boxes, w, h):
  boxes_w = image_boxes[:, 2]
  boxes_h = image_boxes[:, 3]
  converted_boxes = np.stack([
    (image_boxes[:, 0] + boxes_w/2) / w,
    (image_boxes[:, 1] + boxes_h/2) / h,
    boxes_w/w,
    boxes_h/h
  ], -1)
  return converted_boxes


def hflip_xywh_box(box, width):
  box_w = box[:, 2]
  cx = box[:, 0] + box_w/2.0
  cx = width - cx
  box[:, 0] = cx - box_w/2.0


def hflip_cxcywh_box(box, width):
  box[:, 0] = width - box[:, 0]


@dataclass
class ImageCollateWithBoxes(ImageCollater):
  boxes: str
  train_transforms: Dict[str, Any]
  eval_transform: Dict[str, Any]
  is_train: bool
  image_size: Tuple[int, int]
  box_format: str
  horizontal_flip: float
  horizontal_flip_tasks: Any
  return_objectness: bool
  cached_bboxes: Dict
  hdf5_box_format: str = None

  def __post_init__(self):
    if self.hdf5_box_format is None:
      with h5py.File(self.boxes, "r") as f:
        self.hdf5_box_format: str = f["box_format"][()]

  def collate(self, examples):
    boxes = []
    objectness = []
    detr_labels = []
    with h5py.File(self.boxes, "r") as f:
      for example in examples:
        if example.image_id in self.cached_bboxes:
          box, obj = self.cached_bboxes[example.image_id]
          boxes.append(box)
          objectness.append(obj)
          detr_labels.append(obj)
        else:
          example_boxes = f[str(example.image_id)]["bboxes"][:]
          if self.hdf5_box_format != "cxcywh":
            example_boxes = torchvision.ops.box_convert(
              torch.as_tensor(example_boxes), self.hdf5_box_format, "cxcywh").numpy()
          boxes.append(example_boxes)
          if self.return_objectness:
            objectness.append(f[str(example.image_id)]["objectness"][:])
            # objectness.append(f[str(example.image_id)]["relevance_logits"][:])

    image_tensors = []
    image_sizes = []
    for i, example in enumerate(examples):
      if self.is_train:
        trans = self.train_transforms[example.task]
      else:
        trans = self.eval_transform

      img, size = image_utils.load_image_data(example, self.image_size)
      img = trans(img)

      if (
          self.is_train and
          self.horizontal_flip and
          example.task in self.horizontal_flip_tasks and
          self.horizontal_flip and torch.rand(1) < self.horizontal_flip
      ):
        img = hflip(img)
        hflip_cxcywh_box(boxes[i], 1.0)
        if example.target_boxes is not None:
          flipped_targets = np.copy(example.target_boxes)
          hflip_xywh_box(flipped_targets, size[1])
          examples[i] = replace(example, target_boxes=flipped_targets)

      image_tensors.append(img)
      image_sizes.append(size)

    images = nested_tensor_from_tensor_list(image_tensors)

    box_lens = [len(x) for x in boxes]
    if all(box_lens[0] == x for x in box_lens[1:]):
      boxes = torch.as_tensor(boxes, dtype=torch.float32)
      if self.return_objectness:
        objectness = torch.as_tensor(objectness, dtype=torch.float32)
      n_boxes = None
    else:
      n_boxes = torch.as_tensor(box_lens, dtype=torch.long)
      max_len = n_boxes.max()
      boxes = our_utils.stack_and_pad(boxes, max_len)
      # -1000 so to keep the results in valid log probability space
      if self.return_objectness:
        objectness = our_utils.stack_and_pad(objectness, max_len, -1000)

    out = dict(images=images, boxes=boxes, n_boxes=n_boxes)
    if self.return_objectness:
      out["objectness"] = objectness
    box_targets = get_box_targets(examples, image_sizes, self.box_format)
    return out, box_targets


@ROIFeatureExtractor.register("box-embed-feature-extractor")
class BoxEmbedFeatureExtractor(ROIFeatureExtractor):
  """Does ROI pooling to get features for image regions"""

  def __init__(
      self,
      box_coordinate_embed: Optional[Layer] = None,
      pre_rio: Layer = None,
      post_rio: Layer = None,
      return_objectness = True,
      rio_processor: str = "mean",
      box_coordinate_join: str = "concat",
      rio_size=7,
  ):
    super().__init__()
    self.box_coordinate_embed = box_coordinate_embed
    self.pre_rio = pre_rio
    self.post_rio = post_rio
    self.return_objectness = return_objectness
    self.rio_processor = rio_processor
    self.box_coordinate_join = box_coordinate_join
    self.rio_size = rio_size

  def extract_roi(self, features, boxes: torch.Tensor):
    B,C,H,W = features.size()
    _,N,_ = boxes.size()
    # Convert boxes to x1, y1, x2, y2 format
    cx, cy, w, h = torch.split(boxes, [1, 1, 1, 1], -1)
    scaled_boxes = torch.cat([
      W*(cx - 0.5*w),
      H*(cy - 0.5*h),
      W*(cx + 0.5*w),
      H*(cy + 0.5*h)
    ], -1)
    scaled_boxes = torch.unbind(scaled_boxes)
    roi_features = torchvision.ops.roi_align(
      features, scaled_boxes, output_size=self.rio_size, aligned=True)

    if self.rio_processor == "mean":
      roi_features = roi_features.view(B, N, C, -1).mean(-1)
    elif self.rio_processor == "max":
      roi_features = roi_features.view(B, N, C, -1).max(-1)
    else:
      raise NotImplementedError(self.rio_processor)
    return roi_features

  def forward(self, images, boxes) -> ImageRegionFeatures:
    if self.pre_rio is not None:
      images = self.pre_rio(images)

    images, mask = images.decompose()
    if torch.any(mask):
      raise ValueError()

    roi_features = self.extract_roi(images, boxes)

    if self.post_rio is not None:
      roi_features = self.post_rio(roi_features)

    if self.box_coordinate_embed:
      box_embed = self.box_coordinate_embed(boxes)
      if self.box_coordinate_join == "concat":
        roi_features = torch.cat([roi_features, box_embed], -1)
      else:
        raise NotImplementedError(self.box_coordinate_join)

    return roi_features


@ImageFeatureExtractor.register("from-precomputed-boxes")
class FromPrecomputedBoxes(ImageFeatureExtractor):
  """Feature extractor that uses pre-computed boxes"""

  def __init__(
      self,
      box_source,
      backbone: Layer,
      feature_extractor: ROIFeatureExtractor,
      horizontal_flip: float = 0.5,
      include_all_image_box = False,
      return_objectness = True,
      preload_bboxes=True,
      dbg=False,
      hdf5_box_format=None
  ):
    super().__init__()
    self.return_objectness = return_objectness
    self.horizontal_flip = horizontal_flip
    self.backbone = backbone
    self.box_source = box_source
    self.feature_extractor = feature_extractor
    self.include_all_image_box = include_all_image_box
    self.preload_bboxes = preload_bboxes
    self.dbg = dbg
    self.hdf5_box_format = hdf5_box_format

    # Hard coded for now
    self.image_size = (480, 640)
    self.train_transforms = {x: get_train_transforms(x, cls_horizontal_flip=False) for x in Task}
    self.eval_transform = get_eval_transform()
    self._cached_bboxes = {}

  def get_collate(self, is_train=False):
    box_file = get_hdf5_image_file(self.box_source)

    return ImageCollateWithBoxes(
      box_file, self.train_transforms, self.eval_transform, is_train,
      self.image_size, "cxcywh",
      self.horizontal_flip, [Task.CLS, Task.DETECTION],
      self.return_objectness,
      self._cached_bboxes,
      self.dbg,
      self.hdf5_box_format
    )

  def forward(self, images, boxes, objectness=None, n_boxes=None) -> ImageRegionFeatures:
    if self.include_all_image_box:
      all_image = torch.as_tensor([0.5, 0.5, 1.0, 1.0], dtype=boxes.dtype, device=boxes.device)
      all_image = all_image.unsqueeze(0).repeat(boxes.size(0), 1)
      boxes = torch.cat([all_image.unsqueeze(1), boxes], 1)
      if objectness is not None:
        if len(objectness.size()) == 3:
          all_image_rel = torch.full((objectness.size(0), 1, objectness.size(2)), 0.0,
                                     dtype=objectness.dtype, device=objectness.device)
        else:
          all_image_rel = torch.full((objectness.size(0), 1), np.log(0.5),
                                     dtype=objectness.dtype, device=objectness.device)
        objectness = torch.cat([all_image_rel, objectness], 1)

    roi_features = self.feature_extractor(self.backbone(images), boxes)

    return ImageRegionFeatures(boxes, roi_features, objectness, n_boxes)

  def get_pretrained_parameters(self):
    return list(self.backbone.parameters())


class DebugFeaturizer(ImageFeatureExtractor, ImageCollater):

  def __init__(self, n_boxes=4, dim=32):
    super().__init__()
    self.n_boxes = n_boxes
    self.dim = dim

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return self

  def collate(self, batch):
    targets = get_box_targets(batch, [(640, 480) for _ in batch])
    return dict(batch_size=torch.as_tensor(len(batch))), targets

  def forward(self, batch_size) -> ImageRegionFeatures:
    device = batch_size.device
    return ImageRegionFeatures(
      torch.empty(batch_size, self.n_boxes, 4, device=device).uniform_(-1.0, 1.0),
      torch.empty(batch_size, self.n_boxes, self.dim, device=device).uniform_(-1.0, 1.0),
      torch.log(torch.empty(batch_size, self.n_boxes, self.dim, device=device).uniform_(1e-6, 1.0 - 1e-6))
    )


def extract_query_boxes(
    hdf5_group, image_id, query_boxes, box_format, size, extract_features=True):
  # Convert the queries to the format expected in the hdf5 file
  query_boxes = torch.tensor(query_boxes, dtype=torch.float32)
  if torch.any(query_boxes > 1.0):  # If the boxes are not normalized, normalize them
    if size is None:
      size = image_utils.get_image_size(image_id)
    query_boxes /= torch.as_tensor([size[0], size[1], size[0], size[1]])
  query_boxes = torchvision.ops.box_convert(query_boxes, "xywh", box_format)
  saved_boxes = hdf5_group["query_bboxes"][:]

  matches = np.abs(np.expand_dims(query_boxes.numpy(), 1) - np.expand_dims(saved_boxes, 0)) < 1e-6
  matches = np.all(matches, -1)
  q_ixs = []
  for i, box in enumerate(query_boxes):
    match_ix = np.where(matches[i])[0]
    if len(match_ix) != 1:
      print(saved_boxes)
      print(query_boxes)
      raise ValueError(f"Unable to locate a required query box for image {image_id}")
    q_ixs.append(match_ix[0])

  # TODO could avoid loading the entire array
  objectness = torch.as_tensor(hdf5_group["query_objectness"][:][q_ixs])
  if extract_features:
    return query_boxes, objectness, torch.as_tensor(hdf5_group["query_features"][:][q_ixs])
  else:
    return query_boxes, objectness
  # We do some footwork here since hdf5 needs sorted arrays as input, but we want to output
  # boxes that match the input order
  # q_ixs = np.array(q_ixs)
  # arg_sort = q_ixs.argsort()
  # q_ixs_sorted = q_ixs[arg_sort]
  # undo_sort = np.argsort(arg_sort)
  #
  # objectness = torch.as_tensor(hdf5_group["query_objectness"][q_ixs_sorted][undo_sort])
  # if extract_features:
  #   return query_boxes, objectness, torch.as_tensor(hdf5_group["query_features"][q_ixs_sorted][undo_sort])
  # else:
  #   return query_boxes, objectness


@dataclass
class MultiHdf5FeatureExtractorCollate(ImageCollater):
  source_files: List[str]
  box_format: str = None
  output_box_format: str = "cxcywh"
  return_features: bool = True
  return_objectness: bool = True

  def __post_init__(self):
    self.key_to_ix = {}

    for ix, file in enumerate(self.source_files[:-1]):
      logging.info(f"Building key/file map for {file}...")

      with h5py.File(file, "r") as f:
        for key in f.keys():
          self.key_to_ix[key] = ix

    logging.info("Done")

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
    boxes = []
    features = []
    objectness = []
    image_sizes = []

    with ExitStack() as stack:
      files = [stack.enter_context(h5py.File(name, "r")) for name in self.source_files]
      for ex in batch:
        key = image_utils.get_cropped_img_key(ex.image_id, ex.crop)
        ix = self.key_to_ix.get(key, -1)
        grp = files[ix][key]

        if ex.target_boxes is not None:
          image_sizes.append(image_utils.get_image_size(ex.image_id)[::-1])
        else:
          image_sizes.append(None)

        box_key = "bboxes" if 'bboxes' in grp else "boxes"
        image_boxes = torch.as_tensor(grp[box_key][:])

        if self.return_features:
          features.append(torch.as_tensor(grp['features'][:]))
          assert len(features[-1]) == len(image_boxes)
        if self.return_objectness:
          objectness.append(torch.as_tensor(grp['objectness'][:]))
          assert len(objectness[-1]) == len(image_boxes)

        if ex.query_boxes is not None:
          # Load recorded query features/objectness
          # TODO if client does not want objectness or features, could just skip this completely
          out = extract_query_boxes(
            grp, ex.image_id, ex.query_boxes, self.box_format,
            image_sizes[-1], extract_features=self.return_features)
          image_boxes = torch.cat([image_boxes, out[0]], 0)
          if self.return_objectness:
            objectness[-1] = torch.cat([objectness[-1], out[1]], 0)
          if self.return_features:
            features[-1] = torch.cat([features[-1], out[2]], 0)

        image_boxes = torchvision.ops.box_convert(image_boxes, self.box_format, self.output_box_format)
        boxes.append(image_boxes)

    target = get_box_targets(batch, image_sizes)
    n_boxes = [len(x) for x in boxes]
    max_len = max(n_boxes)
    if all(n_boxes[0] == x for x in n_boxes[1:]):
      fe = ImageRegionFeatures(
        torch.stack(boxes),
        torch.stack(features) if self.return_features else None,
        torch.stack(objectness) if self.return_objectness else None,
      )
    else:
      n_boxes = torch.as_tensor(n_boxes, dtype=torch.long)
      fe = ImageRegionFeatures(
        our_utils.stack_and_pad(boxes, max_len),
        our_utils.stack_and_pad(features, max_len) if self.return_features else None,
        (our_utils.stack_and_pad(objectness, max_len, pad=-10000)
         if self.return_objectness else None),
        n_boxes,
      )

    return dict(features=fe), target


@dataclass
class Hdf5FeatureExtractorCollate(ImageCollater):
  source_file: str
  box_format: str = None
  output_box_format: str = "cxcywh"
  return_features: bool = True
  return_objectness: bool = True

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
    boxes = []
    features = []
    objectness = []
    image_sizes = []
    with h5py.File(self.source_file, "r") as f:
      for ex in batch:

        if hasattr(ex.image_id, "load_vinvl_features"):
          # A bit of a hack, but we let some examples short-cut the usual loading
          # processing by having an image_id object rather than a image_id string
          assert ex.query_boxes is None
          f, b, o = ex.image_id.load_vinvl_features()
          features.append(f)
          objectness.append(o)
          boxes.append(b)
          continue

        grp = f[image_utils.get_cropped_img_key(ex.image_id, ex.crop)]

        if ex.target_boxes is not None:
          image_sizes.append(image_utils.get_image_size(ex.image_id)[::-1])
        else:
          image_sizes.append(None)

        box_key = "bboxes" if 'bboxes' in grp else "boxes"
        image_boxes = torch.as_tensor(grp[box_key][:])

        if self.return_features:
          features.append(torch.as_tensor(grp['features'][:]))
        if self.return_objectness:
          objectness.append(torch.as_tensor(grp['objectness'][:]))

        if ex.query_boxes is not None:
          # Load recorded query features/objectness
          # TODO if client does not want objectness or features, could just skip this completely
          out = extract_query_boxes(
            grp, ex.image_id, ex.query_boxes, self.box_format,
            image_sizes[-1], extract_features=self.return_features)
          image_boxes = torch.cat([image_boxes, out[0]], 0)
          if self.return_objectness:
            objectness[-1] = torch.cat([objectness[-1], out[1]], 0)
          if self.return_features:
            features[-1] = torch.cat([features[-1], out[2]], 0)

        image_boxes = torchvision.ops.box_convert(image_boxes, self.box_format, self.output_box_format)
        boxes.append(image_boxes)

    target = get_box_targets(batch, image_sizes)
    n_boxes = [len(x) for x in boxes]
    max_len = max(n_boxes)
    if all(n_boxes[0] == x for x in n_boxes[1:]):
      fe = ImageRegionFeatures(
        torch.stack(boxes),
        torch.stack(features) if self.return_features else None,
        torch.stack(objectness) if self.return_objectness else None,
      )
    else:
      n_boxes = torch.as_tensor(n_boxes, dtype=torch.long)
      fe = ImageRegionFeatures(
        our_utils.stack_and_pad(boxes, max_len),
        our_utils.stack_and_pad(features, max_len) if self.return_features else None,
        (our_utils.stack_and_pad(objectness, max_len, pad=-10000)
         if self.return_objectness else None),
        n_boxes,
      )
    return dict(features=fe), target


@ImageFeatureExtractor.register("hdf5-feature-extractor")
class Hdf5FeatureExtractor(ImageFeatureExtractor):
  """Loadings features from HDF5"""

  @classmethod
  def from_params(
        cls,
        params: Params,
        **extras,
    ):

    # older version of this class stored a list of file, see if we can
    # convert
    source = params["source"]
    if isinstance(source, list):
      if all(x.endswith("vinvl") for x in source):
        params["source"] = "vinvl"
      else:
        raise NotImplementedError(source)
    return super().from_params(params, **extras)

  def __init__(self, source, box_format=None, box_embedder: Layer=None):
    super().__init__()
    self.box_embedder = box_embedder
    self.source = source
    self.box_format = box_format

    # Look up some meta-data in the hdf5 file
    sources = image_utils.get_hdf5_files(source)
    logging.info(f"Found {len(sources)} feature files for {self.source}: {sources}")
    if self.box_format is None:
      _box_formats = []
      for src in sources:
        with h5py.File(src, "r") as f:
          _box_formats.append(f["box_format"][()])
      assert all(_box_formats[0] == x for x in _box_formats[1:])
      self.box_format = _box_formats[0]
    else:
      self._box_format = self.box_format

  def get_collate(self, is_train=False) -> 'ImageCollater':
    sources = image_utils.get_hdf5_files(self.source)
    if len(sources) == 1:
      return Hdf5FeatureExtractorCollate(sources[0], self._box_format)
    else:
      return MultiHdf5FeatureExtractorCollate(sources, self.box_format)

  def forward(self, features) -> ImageRegionFeatures:
    if self.box_embedder:
      box_embed = self.box_embedder(features.boxes)
      features.features = torch.cat([
        features.features,
        box_embed
      ], -1)
    return features

