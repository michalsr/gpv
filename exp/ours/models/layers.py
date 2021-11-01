import logging
import math

import torch
import torchvision
from allennlp.common import Registrable, Params

from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from transformers.models.t5.modeling_t5 import T5LayerNorm

from exp.gpv.models.backbone import Backbone
from exp.gpv.models.position_encoding import PositionEmbeddingSine
from exp.ours import file_paths
from exp.ours.util import py_utils, our_utils
from exp.ours.util.to_params import to_params
from utils.detr_misc import NestedTensor


class Layer(Registrable, nn.Module):
  pass


def get_fn(fn_name):
  if fn_name == "relu":
    return torch.relu
  else:
    raise NotImplementedError()


@Layer.register("linear")
class Linear(nn.Linear, Layer):
  def to_params(self):
    return dict(
      in_features=self.in_features,
      out_features=self.out_features,
      bias=self.bias is not None,
    )


@Layer.register("attenpool")
class AttentionPool2d(Layer):

  def __init__(self, spacial_dim, embed_dim, num_heads, output_dim):
    super().__init__()
    self.spacial_dim = spacial_dim
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.output_dim = output_dim

    self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
    self.k_proj = nn.Linear(embed_dim, embed_dim)
    self.q_proj = nn.Linear(embed_dim, embed_dim)
    self.v_proj = nn.Linear(embed_dim, embed_dim)
    self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)

  def forward(self, x):
    box_embed = x[:, :, -5:]
    x = x[:, :, :-5]
    batch, seq, dim = x.size()
    x = x.view(-1, dim)
    # To [NCHW] format
    x = x.view(batch*seq, self.embed_dim, self.spacial_dim, self.spacial_dim)
    # NCHW -> (HW)NC
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
    x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
    x, _ = F.multi_head_attention_forward(
      query=x, key=x, value=x,
      embed_dim_to_check=x.shape[-1],
      num_heads=self.num_heads,
      q_proj_weight=self.q_proj.weight,
      k_proj_weight=self.k_proj.weight,
      v_proj_weight=self.v_proj.weight,
      in_proj_weight=None,
      in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
      bias_k=None,
      bias_v=None,
      add_zero_attn=False,
      dropout_p=0,
      out_proj_weight=self.c_proj.weight,
      out_proj_bias=self.c_proj.bias,
      use_separate_proj_weight=True,
      training=self.training,
      need_weights=False
    )
    x = x[0]
    # print(x.size())
    x = x.reshape(batch, seq, -1)
    # print(x.size())
    return torch.cat([x, box_embed], -1)



@Layer.register("relu")
class Relu(nn.ReLU, Layer):
  pass


@Layer.register("null")
class NullLayer(Layer):
  def forward(self, x):
    return x


@Layer.register("sequential")
class Sequence(nn.Sequential, Layer):

  @classmethod
  def from_params(
        cls,
        params: Params,
        constructor_to_call = None,
        constructor_to_inspect = None,
        **extras,
  ):
    return Sequence(*[Layer.from_params(x) for x in params["args"]])

  def to_params(self):
    return dict(args=[to_params(x, Layer) for x in self])


@Layer.register("linear-objectness")
class LinearObjectness(Layer):

  def __init__(self, n_in):
    super().__init__()
    self.n_in = n_in
    self.norm = T5LayerNorm(self.n_in)
    self.rel = nn.Linear(self.n_in, 1)

  def forward(self, encoder, objectness, boxes):
    n_images = boxes.size(1)
    image_rel = self.rel(self.norm(encoder[:, :n_images]))
    return image_rel.squeeze(-1)


@Layer.register("sum-with-objectness-v3")
class SumWithObjectness(Layer):

  def __init__(self, n_in, objectness_factor=False, multi_class_mode="any-object"):
    super().__init__()
    self.n_in = n_in
    self.multi_class_mode = multi_class_mode
    self.objectness_factor = objectness_factor
    self.norm = T5LayerNorm(self.n_in)
    self.rel = nn.Linear(self.n_in, 1 + objectness_factor)
    # Initialize so it just uses objectness at the start
    self.rel.weight.data[:] = 0
    if objectness_factor:
      self.rel.bias.data[1] = 1.0

  def forward(self, encoder, objectness, boxes):
    n_images = boxes.size(1)
    image_rel = self.rel(self.norm(encoder[:, :n_images]))
    if self.objectness_factor:
      image_rel, factor = torch.split(image_rel, [1, 1], -1)
      factor = factor.squeeze(-1)
    else:
      factor = None
    image_rel = image_rel.squeeze(-1)

    if len(objectness.size()) == 3:
      if self.multi_class_mode == "any-object":
        non_object_lp = F.log_softmax(objectness, -1)[:, :, -1]
        object_lp = torch.log1p(-torch.exp(non_object_lp))
      elif self.multi_class_mode == "max-object":
        object_lp = torch.max(F.log_softmax(objectness, -1)[:, :, :-1], 2)[0]
        non_object_lp = torch.log1p(-torch.exp(object_lp))
      else:
        raise NotImplementedError()
      objectness = object_lp - non_object_lp
    else:
      # Note we need eps=-1e-6 to stop NaN occurring if the objectness score is too close to log(1) = 0
      # This has occured in (very) rare cases for the VinVL model, in particular for images
      # 8cdae499db22a787a5274d4ee2255315964e1144ab3f95665144c90e24d79917
      # 597caa946a207ef96ede01a53321ff4fdf000a48707ea7c495627330b3ee4b90
      objectness = our_utils.convert_logprob_to_sigmoid_logit(objectness, -1e-6)

    if factor is not None:
      objectness = objectness * factor

    return image_rel + objectness


@Layer.register("sum-with-objectness-v2")
class SumWithObjectnessV2(Layer):

  def __init__(self, n_in, mode="any-object", objectness_factor=False):
    super().__init__()
    self.n_in = n_in
    self.mode = mode
    self.objectness_factor = objectness_factor
    self.norm = T5LayerNorm(self.n_in)
    self.rel = nn.Linear(self.n_in, 1 + objectness_factor)
    # Initialize so it just uses objectness at the start
    self.rel.weight.data[:] = 0
    if objectness_factor:
      self.rel.bias.data[1] = 1.0

  def forward(self, encoder, objectness, boxes):
    n_images = boxes.size(1)
    image_rel = self.rel(self.norm(encoder[:, :n_images]))
    if self.objectness_factor:
      image_rel, factor = torch.split(image_rel, [1, 1], -1)
      factor = factor.squeeze(-1)
    else:
      factor = None
    image_rel = image_rel.squeeze(-1)

    if self.mode == "any-object":
      non_object_lp = F.log_softmax(objectness, -1)[:, :, -1]
      object_lp = torch.log1p(-torch.exp(non_object_lp))
    elif self.mode == "none":
      return image_rel
    elif self.mode == "max-object":
      object_lp = torch.max(F.log_softmax(objectness, -1)[:, :, :-1], 2)[0]
      non_object_lp = torch.log1p(-torch.exp(object_lp))
    else:
      raise NotImplementedError()

    objectness = object_lp - non_object_lp
    if factor is not None:
      objectness = objectness * factor
    return image_rel + objectness


@Layer.register("image-rel-decode-v1")
class ImageRelDecoderV1(Layer):

  def __init__(self, n_in, hidden_dim, n_heads, dim_feedforward,
               n_layers, box_embedder: Layer, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.n_in = n_in
    self.dim_feedforward = dim_feedforward
    self.hidden_dim = hidden_dim
    self.box_embedder = box_embedder
    self.dropout = dropout
    self.n_layers = n_layers

    self.norm = T5LayerNorm(n_in)
    self.join = nn.Linear(n_in, hidden_dim)
    self.cls = nn.Linear(hidden_dim, 1)
    layers = [
      nn.TransformerEncoderLayer(hidden_dim, n_heads, dim_feedforward, dropout)
      for _ in range(n_layers)
    ]
    self.layers = nn.Sequential(*layers)

  def forward(self, encoder, objectness, boxes):
    batch, n_images = objectness.size()[:2]
    embed = self.join(self.norm(encoder[:, :n_images]))
    box_embed = self.box_embedder(boxes)
    for l in self.layers:
      embed = l(embed + box_embed)
    return self.cls(embed).squeeze(-1)


@Layer.register("predict-objectness")
class PredictObjectness(Layer):

  def __init__(self, n_in, n_classes, w):
    super().__init__()
    self.n_classes = n_classes
    self.n_in = n_in
    self.w = w
    self.cls = nn.Linear(n_in, n_classes)

  def forward(self, box_embed, image_features):
    pred = self.cls(box_embed)
    lp = F.log_softmax(pred, -1)
    total_lp = lp*F.softmax(image_features.objectness, -1)
    loss = -total_lp.sum(-1).mean(1).mean()
    return loss*self.w, dict(object_loss=loss)


@Layer.register("basic-box-embedder")
class BasicBoxEmbedder(Layer):
  def forward(self, boxes: torch.Tensor):
    cx, cy, w, h = [x.squeeze(-1) for x in boxes.split([1, 1, 1, 1], -1)]
    return torch.stack([
      cx,
      cy,
      w,
      h,
      w*h
    ], -1)


@Layer.register("with-log")
class NonLinearCoordinateEncoder(Layer):
  def __init__(self, log_alpha, coordinate_layer: Layer=None):
    super().__init__()
    self.log_alpha = log_alpha
    self.coordinate_layer = coordinate_layer
    self.register_buffer("_log_alpha", torch.as_tensor(log_alpha, dtype=torch.float32))

  def forward(self, x):
    batch, seq, dim = x.size()
    alpha = self._log_alpha.view(1, 1, 1, -1)
    logs = torch.cat([
      torch.log(x.unsqueeze(-1) + alpha),
      torch.log1p(- x.unsqueeze(-1) + alpha),
    ], -1)
    logs = logs.view(batch, seq, -1)
    logs = torch.cat([x, logs], -1)
    if torch.any(torch.isnan(logs)):
      print(x.min(), x.max())
      raise ValueError()
    if self.coordinate_layer:
      logs = self.coordinate_layer(logs)
    return logs


@Layer.register("layer-norm")
class LayerNorm(Layer):
  def __init__(self, eps=1e-5):
    super().__init__()
    self.eps = eps

  def forward(self, x):
    return F.layer_norm(x, (x.size(-1),), eps=self.eps)


@Layer.register("resnet-fpn")
class ResnsetFPNBackbone(Layer):
  def __init__(self, name: str, pretrain: bool, trainable_layers: int):
    super().__init__()
    self.name = name
    self.pretrain = pretrain
    self.trainable_layers = trainable_layers
    self.model = resnet_fpn_backbone('resnet18', True, trainable_layers=3)

  def forward(self, *args, **kwargs):
    return self.model(*args, **kwargs)


@Layer.register("detectron-backbone")
class DetectronBackbone(Layer):
  def __init__(self, name="COCO-Detection/faster_rcnn_R_50_C4_1x.yaml", frozenbatchnorm=True, freeze=None):
    super().__init__()
    self.frozenbatchnorm = frozenbatchnorm
    self.freeze = freeze
    self.name = name
    logging.info(f"Loading model {name}")
    with py_utils.DisableLogging():
      # Keep the import here for now so the dependency is optional
      from detectron2 import model_zoo
      self.model = model_zoo.get(name, True).backbone
      self.model.eval()
      print(model_zoo.get_config(name, True).INPUT)
      raise ValueError()

  def forward(self, x):
    if isinstance(x, NestedTensor):
      x = x.tensors
    out = self.model(x)
    assert len(out) == 1
    return list(out.values())[0]


@Layer.register("torchvision-backbone")
@Layer.register("detr-backbone")
class TorchvisionBackbone(Layer):
  def __init__(self, backbone="resnet50", dilation: bool=False, frozenbatchnorm=True,
               freeze=None):
    super().__init__()
    self.backbone = backbone
    self.dilation = dilation
    self.frozenbatchnorm = frozenbatchnorm

    norm_layer = nn.BatchNorm2d
    if frozenbatchnorm:
      norm_layer = FrozenBatchNorm2d

    return_layers = {'layer4': "0"}
    backbone = getattr(torchvision.models, backbone)(
      replace_stride_with_dilation=[False, False, dilation],
      pretrained=True, norm_layer=norm_layer)
    self.model = IntermediateLayerGetter(backbone, return_layers=return_layers)

    self.freeze = freeze

    if self.freeze == "all":
      for p in self.parameters():
        p.requires_grad = False

    elif isinstance(self.freeze, int):
      assert 0 <= self.freeze <= 5
      to_train = 5 - self.freeze
      layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:to_train]
      if to_train == 5:
        layers_to_train.append('bn1')
      for name, parameter in self.model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
          parameter.requires_grad_(False)

    elif self.freeze is not None and self.freeze != "none":
      raise NotImplementedError(self.freeze)

  def forward(self, x) -> NestedTensor:
    if isinstance(x, NestedTensor):
      x = x.tensors
    out = self.model(x)
    assert len(out) == 1
    return out["0"]


@Layer.register("pretrained-detr-backbone")
class PretrainedDetrBackbone(TorchvisionBackbone):
  def __init__(self, detr_model, freeze=None):
    super().__init__(freeze=freeze)
    self.detr_model = detr_model
    state_dict = torch.load(file_paths.PRETRAINED_DETR_MODELS[detr_model], map_location="cpu")["model"]
    tmp = py_utils.extract_module_from_state_dict(state_dict, "backbone.0.body")
    self.model.load_state_dict(tmp)


class AddSinePositionalEmbedding(Layer):
  def __init__(self, num_features=128, temperature=10000, normalize=False):
    super().__init__()
    self.num_features = num_features
    self.temperature = temperature
    self.normalize = normalize
    self.pos_embedder = PositionEmbeddingSine(num_features, temperature, normalize=normalize)

  def forward(self, x: NestedTensor):
    pos = self.pos_embedder(x)
    return NestedTensor(x+pos, x.mask)


@Layer.register("two-layer-ff")
class TwoLayerFF(Layer):

  def __init__(self, input_dim, hidden_dim, out_dim, fn="relu", dropout=0.0):
    super().__init__()
    self.out_dim = out_dim
    self.dropout = dropout
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.fn = fn
    self._fn = get_fn(fn)
    self.lin = nn.Linear(input_dim, hidden_dim)
    self.out = nn.Linear(hidden_dim, self.out_dim)

  def forward(self, x):
    out = self._fn(self.lin(x))
    out = torch.dropout(out, self.dropout, self.training)
    return self.out(out)

