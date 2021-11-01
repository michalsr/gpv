from typing import List, Any, Dict, Tuple

import torch
import torchvision

from exp.ours.data.gpv_example import GPVExample
from exp.ours.image_featurizer.image_featurizer import ImageCollater, BoxTargets, \
  ImageRegionFeatures, ImageFeatureExtractor
from exp.ours.util import our_utils, py_utils


class ConceptualCaptionCollate(ImageCollater):

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], BoxTargets]:
    out = []
    for ex in batch:
      out.append(ex.image_id.load_all())
    features, boxes, obj = py_utils.transpose_lists(out)
    fe = ImageRegionFeatures.build_from_lists(boxes, features, obj)
    return dict(features=fe), [None]*len(batch)


@ImageFeatureExtractor.register("cc-feature-extractor")
class ConceptualCaptionFeatureExtractor(ImageFeatureExtractor):

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return ConceptualCaptionCollate()

  def forward(self, features) -> ImageRegionFeatures:
    return features

