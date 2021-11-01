import logging
from contextlib import ExitStack
from os.path import join, exists
from typing import List, Dict, Any, Tuple, Optional, NewType, Union

import logging
from typing import List, Dict, Any, Tuple, Optional

import h5py
import numpy as np
import torch
import torchvision
from allennlp.common import Registrable, Params, FromParams
from dataclasses import dataclass, replace
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import ColorJitter, RandomGrayscale, Normalize
from torchvision.transforms.functional import hflip, to_tensor
from torch.nn import functional as F

from exp.ours import file_paths
from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.dataset import Task
from exp.ours.models.gpv1_preprocessing import get_train_transforms, get_eval_transform
from exp.ours.models.layers import Layer
from exp.ours.util import image_utils, py_utils, our_utils
from exp.ours.util.image_utils import get_hdf5_image_file, get_hdf5_files
from exp.vinvl.structures.image_list import to_image_list
from exp.vinvl.transforms import RandomHorizontalFlip, Compose
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
      None if features is None else our_utils.stack_and_pad(features, max_len),
      # -10000 so the padding is a valid log-probability
      None if objectness is None else our_utils.stack_and_pad(objectness, max_len, -10000),
      n_boxes,
    )

  boxes: torch.Tensor
  """[batch, n_regions, 4] boxes in [cx, cy, w, y] format normalized between 0 and 1"""

  features: Optional[torch.Tensor]
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

  def get_n_boxes(self):
    if self.n_boxes is None:
      batch, n = self.boxes.size()[:2]
      return torch.full((batch,), n,
                        device=self.boxes.device, dtype=torch.long)
    else:
      return self.n_boxes


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


def gather_qboxes_and_targets(batch, hflipped=None, qbox_format="cxcywh"):
  """Utility method that gather the query_boxes and targets of a batch"""
  targets = []
  query_boxes = []
  if hflipped is None:
    hflipped = [None for _ in batch]
  for ex, is_flipped in zip(batch, hflipped):
    if ex.target_boxes is None:
      targets.append(None)
    else:
      if ex.crop:
        raise ValueError("Box target on cropped images no supoorted")
      # Normalize the target boxes to be between 0 and 1 and to be
      # cxcywh format
      # TODO it would be nice to do this in the pre-preprocesing step
      boxes = torch.tensor(ex.target_boxes, dtype=torch.float)
      boxes = torchvision.ops.box_convert(boxes, "xywh", "cxcywh")
      boxes = image_utils.normalize_boxes(boxes, ex.image_id)
      if is_flipped:
        boxes[:, 0] = 1.0 - boxes[:, 0]
      targets.append(boxes)

    if ex.query_boxes is None:
      query_boxes.append(None)
    else:
      # Convert query boxes
      qboxes = torch.tensor(ex.query_boxes, dtype=torch.float)
      if torch.any(qboxes > 1.0):
        qboxes = image_utils.normalize_boxes(qboxes, ex.image_id)
      qboxes = torchvision.ops.box_convert(qboxes, "xywh", qbox_format)
      if is_flipped:
        qboxes[:, 0] = 1.0 - qboxes[:, 0]
      query_boxes.append(qboxes)
  return query_boxes, targets


class PrecomputedDataLoader:
  """Utility class that gathers pre-computed data and targets of a batch"""

  def __init__(self, box_sources, extract_features=False, extract_objectness=True):
    self.box_sources = box_sources
    self.extract_objectness = extract_objectness
    self.extract_features = extract_features
    self.key_to_ix = {}
    if self.box_sources != "debug":
      for ix, file in enumerate(self.box_sources):
        logging.info(f"Building key/file map for {file}...")
        with h5py.File(file, "r") as f:
          for key in f.keys():
            self.key_to_ix[key] = ix

  def __call__(self, batch: List[GPVExample], hflipped=None) -> Tuple[ImageRegionFeatures, BoxTargets]:
    query_boxes, targets = gather_qboxes_and_targets(batch, qbox_format="xyxy")
    if hflipped is None:
      hflipped = [None for _ in batch]

    if self.box_sources == "debug":
      batch_size = len(batch)
      n_boxes = 50
      boxes = torch.empty(batch_size, n_boxes, 4).uniform_(0.00001, 0.5)
      for i, q in enumerate(query_boxes):
        if q is not None:
          boxes[i, :q.shape[0]] = torch.as_tensor(q)
      return ImageRegionFeatures(
        boxes, None,
        torch.log(torch.empty(batch_size, n_boxes).uniform_(1e-6, 1.0 - 1e-6))
      ), targets

    # For backwards compat
    # TODO remove when no longer needed
    extract_objectness = getattr(self, "extract_objectness", True)

    # Additionally load the boxes
    with ExitStack() as stack:
      all_features = []
      all_bboxes = []
      all_objectness = []
      files = [stack.enter_context(h5py.File(name, "r")) for name in self.box_sources]
      for ex, is_flipped, qboxes in zip(batch, hflipped, query_boxes):
        key = image_utils.get_cropped_img_key(ex.image_id, ex.crop)
        ix = self.key_to_ix.get(key, -1)
        grp = files[ix][key]
        bboxes = torch.as_tensor(grp["bboxes"][:])

        if extract_objectness:
          objectness = torch.as_tensor(grp['objectness'][:])

        if self.extract_features:
          features = torch.as_tensor(grp['features'][:])
        if qboxes is not None:
          bboxes = torch.cat([bboxes, qboxes], 0)
          if self.extract_features:
            qobj, qfeatures = find_query_boxes(grp, qboxes, extract_features=True)
            features = torch.cat([features, qfeatures], 0)
          else:
            qobj = find_query_boxes(grp, qboxes, extract_features=False)
          if extract_objectness:
            objectness = torch.cat([objectness, qobj], 0)
        all_bboxes.append(torchvision.ops.box_convert(bboxes, "xyxy", "cxcywh"))
        if extract_objectness:
          all_objectness.append(objectness)
        if self.extract_features:
          all_features.append(features)

    for i, flip in enumerate(hflipped):
      if flip:
        # TODO flip target and all_bboxes
        raise NotImplementedError()

    all_features = all_features if self.extract_features else None
    all_objectness = all_objectness if extract_objectness else None
    regions = ImageRegionFeatures.build_from_lists(all_bboxes, all_features, all_objectness)
    return regions, targets


class ROIFeatureExtractor(Layer):
  """Extract image features for a given set of regions"""

  def forward(self, x: torch.Tensor, boxes: torch.Tensor):
    """
    x: Tensor of images
    boxes: [batch, n, 4] boxes that NOT normalized and in xyxy format
    """
    raise NotImplementedError()


@ROIFeatureExtractor.register("box-embed-feature-extractor")
class BoBoxEmbedFeatureExtractor(ROIFeatureExtractor):
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
    B, C, W, H = features.size()
    N = boxes.size(1)

    div = torch.as_tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
    scaled_boxes = boxes * div
    scaled_boxes = torchvision.ops.box_convert(scaled_boxes, "cxcywh", "xyxy")
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


def build_scaled_boxes(features, boxes):
  B, C, H, W = features.size()
  div = torch.as_tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
  boxes = boxes * div
  return torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")


@ImageFeatureExtractor.register("debug")
class DebugFeaturizer(ImageFeatureExtractor, ImageCollater):

  def __init__(self, n_boxes=4, dim=32):
    super().__init__()
    self.n_boxes = n_boxes
    self.dim = dim

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return self

  def collate(self, batch):
    _, targets = gather_qboxes_and_targets(batch)
    return dict(batch_size=torch.as_tensor(len(batch))), targets

  def forward(self, batch_size) -> ImageRegionFeatures:
    device = batch_size.device
    return ImageRegionFeatures(
      torch.empty(batch_size, self.n_boxes, 4, device=device).uniform_(0.00001, 0.5),
      torch.empty(batch_size, self.n_boxes, self.dim, device=device).uniform_(0.00001, 0.5),
      torch.log(torch.empty(batch_size, self.n_boxes, self.dim, device=device).uniform_(1e-6, 1.0 - 1e-6))
    )


def find_query_boxes(hdf5_group, query_boxes, image_id=None, extract_features=True):
  saved_boxes = hdf5_group["query_bboxes"][:]

  matches = np.abs(np.expand_dims(query_boxes.numpy(), 1) - np.expand_dims(saved_boxes, 0)) < 1e-6
  matches = np.all(matches, -1)
  q_ixs = []
  for i, box in enumerate(query_boxes):
    match_ix = np.where(matches[i])[0]
    if len(match_ix) != 1:
      if image_id is None:
        print(query_boxes)
        print(saved_boxes)
        raise ValueError(f"Unable to locate a required query box")
      else:
        print(query_boxes)
        print(saved_boxes)
        raise ValueError(f"Unable to locate a required query box for image {image_id}")
    q_ixs.append(match_ix[0])

  # TODO could avoid loading the entire array
  objectness = torch.as_tensor(hdf5_group["query_objectness"][:][q_ixs])
  if extract_features:
    return objectness, torch.as_tensor(hdf5_group["query_features"][:][q_ixs])
  else:
    return objectness
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

########### VERSION 2 ############


# def get_box_targets(examples: List[GPVExample], image_sizes, box_format="cxcywh") -> BoxTargets:
#   all_boxes = []
#   for i, example in enumerate(examples):
#     if example.target_boxes is None:
#       all_boxes.append(None)
#     else:
#       boxes = torch.as_tensor(example.target_boxes, dtype=torch.float)
#       x, y, w, h = boxes.split([1, 1, 1, 1], 1)
#       if box_format == "xyxy":
#         box_parts = [x, y, x + w, y + h]
#       elif box_format == "cxcywh":
#         box_parts = [x + 0.5*w, y + 0.5*h, w, h]
#       else:
#         raise NotImplementedError(box_format)
#       if image_sizes is not None:
#         H, W = image_sizes[i]
#         boxes = torch.cat([
#           box_parts[0] / W, box_parts[1] / H,
#           box_parts[2] / W, box_parts[3] / H
#         ], 1)
#       else:
#         boxes = torch.cat([box_parts[0], box_parts[1], box_parts[2], box_parts[3]], 1)
#       all_boxes.append(boxes)
#   return all_boxes
#
#
# def extract_query_boxes(
#     hdf5_group, image_id, query_boxes, box_format, size, extract_features=True):
#   # Convert the queries to the format expected in the hdf5 file
#   query_boxes = torch.tensor(query_boxes, dtype=torch.float32)
#   if torch.any(query_boxes > 1.0):  # If the boxes are not normalized, normalize them
#     if size is None:
#       size = image_utils.get_image_size(image_id)
#     query_boxes /= torch.as_tensor([size[0], size[1], size[0], size[1]])
#   query_boxes = torchvision.ops.box_convert(query_boxes, "xywh", box_format)
#   saved_boxes = hdf5_group["query_bboxes"][:]
#
#   matches = np.abs(np.expand_dims(query_boxes.numpy(), 1) - np.expand_dims(saved_boxes, 0)) < 1e-6
#   matches = np.all(matches, -1)
#   q_ixs = []
#   for i, box in enumerate(query_boxes):
#     match_ix = np.where(matches[i])[0]
#     if len(match_ix) != 1:
#       print(saved_boxes)
#       print(query_boxes)
#       raise ValueError(f"Unable to locate a required query box for image {image_id}")
#     q_ixs.append(match_ix[0])
#
#   # TODO could avoid loading the entire array
#   objectness = torch.as_tensor(hdf5_group["query_objectness"][:][q_ixs])
#   if extract_features:
#     return query_boxes, objectness, torch.as_tensor(hdf5_group["query_features"][:][q_ixs])
#   else:
#     return query_boxes, objectness
#
#
# class Hdf5FeatureExtractorCollate:
#   pass
#
#
# @dataclass
# class MultiHdf5FeatureExtractorCollate(ImageCollater):
#   source_files: List[str]
#   box_format: str = None
#   output_box_format: str = "cxcywh"
#   return_features: bool = True
#   return_objectness: bool = True
#
#   def __post_init__(self):
#     self.key_to_ix = {}
#
#     for ix, file in enumerate(self.source_files):
#       logging.info(f"Building key/file map for {file}...")
#
#       with h5py.File(file, "r") as f:
#         for key in f.keys():
#           self.key_to_ix[key] = ix
#
#     logging.info("Done")
#
#   def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
#     boxes = []
#     features = []
#     objectness = []
#     image_sizes = []
#
#     with ExitStack() as stack:
#       files = [stack.enter_context(h5py.File(name, "r")) for name in self.source_files]
#       for ex in batch:
#         key = image_utils.get_cropped_img_key(ex.image_id, ex.crop)
#         ix = self.key_to_ix.get(key, -1)
#         grp = files[ix][key]
#
#         if ex.target_boxes is not None:
#           image_sizes.append(image_utils.get_image_size(ex.image_id)[::-1])
#         else:
#           image_sizes.append(None)
#
#         box_key = "bboxes" if 'bboxes' in grp else "boxes"
#         image_boxes = torch.as_tensor(grp[box_key][:])
#
#         if self.return_features:
#           features.append(torch.as_tensor(grp['features'][:]))
#           assert len(features[-1]) == len(image_boxes)
#         if self.return_objectness:
#           objectness.append(torch.as_tensor(grp['objectness'][:]))
#           assert len(objectness[-1]) == len(image_boxes)
#
#         if ex.query_boxes is not None:
#           # Load recorded query features/objectness
#           # TODO if client does not want objectness or features, could just skip this completely
#           out = extract_query_boxes(
#             grp, ex.image_id, ex.query_boxes, self.box_format,
#             image_sizes[-1], extract_features=self.return_features)
#           image_boxes = torch.cat([image_boxes, out[0]], 0)
#           if self.return_objectness:
#             objectness[-1] = torch.cat([objectness[-1], out[1]], 0)
#           if self.return_features:
#             features[-1] = torch.cat([features[-1], out[2]], 0)
#
#         image_boxes = torchvision.ops.box_convert(image_boxes, self.box_format, self.output_box_format)
#         boxes.append(image_boxes)
#
#     target = get_box_targets(batch, image_sizes)
#     n_boxes = [len(x) for x in boxes]
#     max_len = max(n_boxes)
#     if all(n_boxes[0] == x for x in n_boxes[1:]):
#       fe = ImageRegionFeatures(
#         torch.stack(boxes),
#         torch.stack(features) if self.return_features else None,
#         torch.stack(objectness) if self.return_objectness else None,
#       )
#     else:
#       n_boxes = torch.as_tensor(n_boxes, dtype=torch.long)
#       fe = ImageRegionFeatures(
#         our_utils.stack_and_pad(boxes, max_len),
#         our_utils.stack_and_pad(features, max_len) if self.return_features else None,
#         (our_utils.stack_and_pad(objectness, max_len, pad=-10000)
#          if self.return_objectness else None),
#         n_boxes,
#       )
#     return dict(features=fe), target
#
#
# @ImageFeatureExtractor.register("hdf5-feature-extractor")
# class Hdf5FeatureExtractor(ImageFeatureExtractor):
#   """Loadings features from HDF5"""
#
#   @classmethod
#   def from_params(
#       cls,
#       params: Params,
#       **extras,
#   ):
#
#     # older version of this class stored a list of file, see if we can
#     # convert
#     source = params["source"]
#     if isinstance(source, list):
#       if all(x.endswith("vinvl") for x in source):
#         params["source"] = "vinvl"
#       else:
#         raise NotImplementedError(source)
#     return super().from_params(params, **extras)
#
#   def __init__(self, source, box_format=None, box_embedder: Layer=None):
#     super().__init__()
#     self.box_embedder = box_embedder
#     self.source = source
#     self.box_format = box_format
#
#     # Look up some meta-data in the hdf5 file
#     sources = image_utils.get_hdf5_files(source)
#     if len(sources) == 0:
#       raise ValueError()
#     logging.info(f"Found {len(sources)} feature files for {self.source}: {sources}")
#     if self.box_format is None:
#       self._box_format = "xyxy"
#     else:
#       self._box_format = self.box_format
#
#   def get_collate(self, is_train=False) -> 'ImageCollater':
#     # print("DEBUG")
#     sources = image_utils.get_hdf5_files(self.source)
#     if len(sources) == 1 and self.source != "vinvl-dbg":
#       return Hdf5FeatureExtractorCollate(sources[0], self._box_format)
#     else:
#       return MultiHdf5FeatureExtractorCollate(sources, self._box_format)
#
#   def forward(self, features) -> ImageRegionFeatures:
#     if self.box_embedder:
#       box_embed = self.box_embedder(features.boxes)
#       features.features = torch.cat([
#         features.features,
#         box_embed
#       ], -1)
#     return features


########### VERSION 1 ###################


@dataclass
class MultiHdf5FeatureExtractorCollate2(ImageCollater):
  source: PrecomputedDataLoader

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], List]:
    regions, targets = self.source(batch)
    return dict(regions=regions), targets




@ImageFeatureExtractor.register("hdf5-feature-extractor")
# @ImageFeatureExtractor.register("hdf5-feature-extractor")
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
    else:
      params["source"] = source

    if "box_format" in params:
      box_format = params.pop("box_format")
      assert box_format is None or box_format == "xyxy"
    return super().from_params(params, **extras)

  def __init__(self, source, box_embedder: Layer=None, extract_objectness=True):
    super().__init__()
    self.box_embedder = box_embedder
    self.extract_objectness = extract_objectness
    self.source = source
    self.extractor = PrecomputedDataLoader(image_utils.get_hdf5_files(self.source), True,
                                           extract_objectness=extract_objectness)

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return MultiHdf5FeatureExtractorCollate2(self.extractor)

  def forward(self, regions) -> ImageRegionFeatures:
    if self.box_embedder:
      box_embed = self.box_embedder(regions.boxes)
      regions.features = torch.cat([
        regions.features,
        box_embed
      ], -1)
    return regions

