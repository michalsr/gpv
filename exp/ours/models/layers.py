import torch
import torchvision.ops
import transformers.models.deberta.modeling_deberta
from allennlp.common import Registrable, Params
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.models.t5.modeling_t5 import T5LayerNorm

from exp.gpv.models.backbone import build_backbone, Backbone
from exp.gpv.models.position_encoding import build_position_encoding, PositionEmbeddingLearned, \
  PositionEmbeddingSine
from exp.ours import file_paths
from exp.ours.data.source_data import ID_TO_COCO_CATEGORY
from exp.ours.util import our_utils, py_utils
from exp.ours.util.our_utils import replace_params_with_buffers
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


@Layer.register("embedding-simplex")
class EmbeddingSimplex(Layer):
  def __init__(self, n_in, n_embeddings, n_out, init="zero",
               pos_init=1, neg_init=-1, lin_factor=1.0):
    super().__init__()
    self.n_in = n_in
    self.n_embeddings = n_embeddings
    self.n_out = n_out
    self.init = init
    self.pos_init = pos_init
    self.neg_init = neg_init
    self.lin_factor = lin_factor

    self.embed_predictor = nn.Linear(self.n_in, self.n_embeddings)
    self.embed_predictor.weight.data[:] = 0.0
    self.embed_predictor.bias.data[:] = 0
    self.to_add = nn.Linear(self.n_in, self.n_out)
    self.to_add.bias.data[:] = 0
    self.to_add.weight.data[:] = 0
    if self.init == "t5-coco-categories":
      tok = AutoTokenizer.from_pretrained("t5-base")
      ids = set()
      ids.add(tok.eos_token_id)
      for word in ID_TO_COCO_CATEGORY.values():
        ids.update(tok.encode(word))
      self.embed_predictor.bias.data[:] = self.neg_init
      self.embed_predictor.bias.data[list(ids)] = self.pos_init
    elif self.init != "zero":
      raise ValueError()

  def forward(self, features, embeddings):
    embed_score = F.softmax(self.embed_predictor(features), -1)
    embed_features = torch.matmul(embed_score, embeddings.weight[:self.n_embeddings])
    return embed_features + self.to_add(features) * self.lin_factor


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
    else:
      object_lp = objectness
      non_object_lp = torch.log1p(-torch.exp(object_lp))

    objectness = object_lp - non_object_lp
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
    return F.layer_norm(x, (x.size(-1),))


@Layer.register("detr-backbone")
class DetrBackbone(Layer):
  def __init__(self, backbone="resnet50", dilation: bool=False, frozenbatchnorm=True,
               freeze=None):
    super().__init__()
    self.backbone = backbone
    self.dilation = dilation
    self.frozenbatchnorm = frozenbatchnorm
    self.model = Backbone(backbone, True, False, dilation, frozenbatchnorm)
    self.freeze = freeze
    if self.freeze == "all":
      for p in self.parameters():
        p.requires_grad = False
    elif self.freeze is not None and self.freeze != "none":
      raise NotImplementedError(self.freeze)

  def forward(self, x) -> NestedTensor:
    return self.model(x)['0']


@Layer.register("pretrained-detr-backbone")
class PretrainedDetrBackbone(DetrBackbone):
  def __init__(self, detr_model, freeze=None):
    super().__init__(freeze=freeze)
    self.detr_model = detr_model
    state_dict = torch.load(file_paths.PRETRAINED_DETR_MODELS[detr_model], map_location="cpu")["model"]
    tmp = py_utils.extract_module_from_state_dict(state_dict, "backbone.0")
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

