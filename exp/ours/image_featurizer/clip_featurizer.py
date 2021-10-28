import logging
from typing import Any, Tuple, Dict, List, Optional

import torch
import torchvision
from allennlp.common import FromParams
from dataclasses import replace, dataclass

from exp.ours.data.gpv_example import GPVExample
from exp.ours.image_featurizer.image_featurizer import ImageCollater, BoxTargets, \
  gather_qboxes_and_targets, ImageFeatureExtractor, ImageRegionFeatures, PrecomputedDataLoader
from exp.ours.models.layers import Layer
from exp.ours.util import image_utils
import torch.nn.functional as F


class ClipCollater(ImageCollater):
  def __init__(self, transforms, box_extractor=None):
    super().__init__()
    self.transforms = transforms
    self.box_extractor = box_extractor

  def collate(self, batch: List[GPVExample]) -> Tuple[Dict[str, Any], BoxTargets]:
    image_tensors = []
    for example in batch:
      if isinstance(self.transforms, dict):
        trans = self.transforms[example.task]
      else:
        trans = self.transforms
      img = image_utils.load_image_pil(example.image_id, example.crop)
      image_tensors.append(trans(img))
    images = torch.stack(image_tensors, 0)

    if self.box_extractor is None:
      qboxes, targets = gather_qboxes_and_targets(batch)
      return dict(images=images, query_boxes=qboxes), targets
    else:
      regions, targets = self.box_extractor(batch)
      return dict(images=images, regions=regions), targets


@dataclass
class RoIFeatureSource(FromParams):
  layer: int
  output_size: Tuple[int, int]
  sampling_ratio: int = -1
  aggregate: str = "mean"


@ImageFeatureExtractor.register("clip")
class ClipFeaturizer(ImageFeatureExtractor):

  def __init__(self, name, box_source, region_features: List[RoIFeatureSource], freeze="none",
               train_transforms=None, coordinate_embedder: Optional[Layer]=None,
               ):
    super().__init__()
    self.name = name
    self.freeze = freeze
    self.train_transforms = train_transforms
    self.box_source = box_source
    self.region_features = region_features
    self.coordinate_embedder = coordinate_embedder
    if self.train_transforms is not None:
      raise NotImplementedError()

    import clip
    logging.info(f"Loading clip model {self.name}...")
    model, preprocess = clip.load(self.name, "cpu")

    self.visual = model.visual
    self.preprocess = preprocess
    self.box_extractor = PrecomputedDataLoader(image_utils.get_hdf5_files(self.box_source))

    apool = self.visual.attnpool
    self.positional_embedding = apool.positional_embedding
    delattr(apool, "positional_embedding")
    self.visual.positional_embedding = torch.zeros(
      (1, 1), device=self.positional_embedding.device,
      dtype=self.positional_embedding.dtype)

    if self.freeze == "all":
      for param in self.parameters():
        param.requires_grad = False

    elif self.freeze == "conv":
      v = self.visual
      for conv, bn in [(v.conv1, v.bn1), (v.conv2, v.bn2), (v.conv3, v.bn3)]:
        for param in conv.parameters():
          param.requires_grad = False

    elif self.freeze == "non-pool":
      skipped = set()
      for name, param in self.named_parameters():
        if name.startswith("visual.attnpool") or name == "positional_embedding":
          param.requires_grad = True
          skipped.add(name)
        else:
          param.requires_grad = False

      print(list(skipped))
    else:
      raise ValueError()

  def get_collate(self, is_train=False) -> 'ImageCollater':
    return ClipCollater(self.preprocess, self.box_extractor)

  def forward(self, images, regions: ImageRegionFeatures) -> ImageRegionFeatures:
    v = self.visual
    x = images
    for conv, bn in [(v.conv1, v.bn1), (v.conv2, v.bn2), (v.conv3, v.bn3)]:
      x = torch.relu_(bn(conv(x)))

    layer_outs = []
    for layer in [v.layer1, v.layer2, v.layer3, v.layer4]:
      x = layer(x)
      layer_outs.append(x)

    scaled_boxes = torchvision.ops.box_convert(regions.boxes, "cxcywh", "xyxy")
    scaled_boxes = torch.unbind(scaled_boxes)

    all_features = []
    batch_size, n_boxes = regions.boxes.size()[:2]
    for src in self.region_features:
      fe = layer_outs[src.layer]
      h, w = fe.size()[-2:]
      assert h == w
      features = torchvision.ops.roi_align(
        fe, scaled_boxes, src.output_size,
        sampling_ratio=src.sampling_ratio,
        aligned=True, spatial_scale=h
      )
      if src.aggregate == "mean":
        features = features.mean([2, 3]).view(batch_size, -1)
      elif src.aggregate == "max":
        features = features.view(features.size(0), features.size(1), -1)
        features = features.max(-1)[0]
      elif src.aggregate == "atten-pool":
        spacial_dim = self.visual.input_resolution // 32
        apool = self.visual.attnpool
        tmp = self.positional_embedding[:-1].transpose(0, 1).view(-1, spacial_dim, spacial_dim)
        tmp = torchvision.ops.roi_align(
          tmp.unsqueeze(0).expand(fe.size(0), -1, -1, -1), scaled_boxes, src.output_size,
          sampling_ratio=src.sampling_ratio,
          aligned=True, spatial_scale=h
        )
        x = features
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], tmp.shape[2] * tmp.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x += torch.cat([tmp, tmp[-1:]])

        x, _ = F.multi_head_attention_forward(
          query=x, key=x, value=x,
          embed_dim_to_check=x.shape[-1],
          num_heads=apool.num_heads,
          q_proj_weight=apool.q_proj.weight,
          k_proj_weight=apool.k_proj.weight,
          v_proj_weight=apool.v_proj.weight,
          in_proj_weight=None,
          in_proj_bias=torch.cat([apool.q_proj.bias, apool.k_proj.bias, apool.v_proj.bias]),
          bias_k=None,
          bias_v=None,
          add_zero_attn=False,
          dropout_p=0,
          out_proj_weight=apool.c_proj.weight,
          out_proj_bias=apool.c_proj.bias,
          use_separate_proj_weight=True,
          training=apool.training,
          need_weights=False
        )
        features = x[0]
      elif src.aggregate == "max-mean":
        features = features.view(features.size(0), features.size(1), -1)
        features = torch.cat([
          features.mean(-1),
          features.max(-1)[0]
        ], -1)
      elif src.aggregate == "flatten":
        features = features.view(batch_size, -1)
      else:
        raise NotImplementedError(src.aggregate)
      features = features.view(batch_size, n_boxes, -1)
      all_features.append(features)

    if self.coordinate_embedder is not None:
      box_embed = self.coordinate_embedder(regions.boxes)
      all_features.append(box_embed)

    all_features = torch.cat(all_features, -1)
    return replace(regions, features=all_features)
