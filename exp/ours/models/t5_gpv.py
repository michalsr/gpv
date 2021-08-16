import logging
from typing import Union, Tuple, Dict, Optional

import torch
import torchvision.ops
from allennlp.common import Params
from transformers import AutoTokenizer, AutoConfig

from exp.ours.data.gpv import ID_TO_COCO_CATEGORY
from exp.ours.models.model import GPVModel, build_per_example_output
from exp.ours.models.gpv1_preprocessing import Gpv1Preprocessor
from exp.ours.image_featurizer.image_featurizer import ImageFeatureExtractor, ImageRegionFeatures
from exp.ours.models.layers import Layer, Linear
from exp.ours.models.losses import GPVLoss, BasicGPVLoss
import numpy as np

from exp.ours.models.model_utils import CollateWithTokenizer
from exp.ours.models.t5_custom import OurT5ForConditionalGeneration
from exp.ours.train.runner import BeamSearchSpec
from exp.ours.util import our_utils
from exp.ours.util.nlp_utils import encode_with_cache, t5_initialize_decoding
from torch import nn
from torch.nn import functional as F
from exp.ours.util.to_params import to_params


def _fix_old_params(params: Params):
  if "loss" not in params:
    params["loss"] = to_params(BasicGPVLoss(
      1.0, None, None, None, None, None, None, None, sum_seq_tokens=False), GPVLoss)

  if params.pop("aux_obj_cls") is not None:
    raise ValueError()
  if not params.pop("image_first"):
    raise ValueError()
  if "sort_image_features" in params:
    if params.pop("sort_image_features") is not None:
      raise ValueError()
  if params.pop("use_cached_features"):
    raise ValueError()

  if "detr_joiner" in params:
    params["image_joiner"] = params.pop("detr_joiner")
  else:
    params["image_joiner"] = dict(
      type="linear", in_features=2304, out_features=768)
  freeze = params.pop("freeze_detr")
  if freeze is True:
    logging.warning("Freeze set but is not longer supported")
  params["image_feature_extractor"] = {'type': 'detr'}


def _fix_old_state_dict(state_dict: Dict[str, torch.Tensor]):
  for k in list(state_dict):
    if k.startswith("detr_joiner."):
      v = state_dict[k]
      del state_dict[k]
      state_dict["image_joiner." + k[len("detr_joiner."):]] = v

    if k.startswith("detr."):
      v = state_dict[k]
      del state_dict[k]
      state_dict["image_feature_extractor." + k] = v


@GPVModel.register("t5-gpv")
class T5GPV(GPVModel):
  IMAGE_SEP = "[IMAGE]"

  @classmethod
  def from_params(
    cls,
    params: Params,
    **kwargs,
  ):
    if "image_feature_extractor" not in params:
      _fix_old_params(params)
    if "box_aux_loss" in params:
      assert params.pop("box_aux_loss") is None
    if "nms" in params and params["nms"] == 0.0:
      params["nms"] = None
    return super().from_params(params, **kwargs)

  def __init__(
      self,
      t5_model_name: str,
      loss: GPVLoss,
      image_feature_extractor: ImageFeatureExtractor,
      image_joiner: Layer,
      image_relevance: Optional[Layer]=None,
      initialize_t5=True, pre_tokenize=True,
      query_box=None,
      predict_trailing_pad_tokens=False,
      image_positional_bias="zero",
      image_seperator=False,
      freeze_embed=False,
      initialize_joiner="coco",
      nms: float=None,
      all_lower_case=False
  ):
    super().__init__()
    if nms == 0.0:
      # While this is technically possible, it doesn't really make sense to do nms with a 0.0
      # threshold and some old code conflated nms=0.0 with nms=None so we disallow nms=0.0 here
      # so we can treat 0.0 as None when loading this from a parameter file.
      raise ValueError("Need nms > 0.0")
    if freeze_embed and image_seperator:
      raise NotImplementedError()
    self.all_lower_case = all_lower_case
    self.image_relevance = image_relevance
    self.freeze_embed = freeze_embed
    self.t5_model_name = t5_model_name
    self.loss = loss
    self.image_feature_extractor = image_feature_extractor
    self.initialize_t5 = initialize_t5
    self.predict_trailing_pad_tokens = predict_trailing_pad_tokens
    self.image_positional_bias = image_positional_bias
    self.pre_tokenize = pre_tokenize
    self.image_seperator = image_seperator
    self.initialize_joiner = initialize_joiner
    self.initialize_t5 = initialize_t5
    self.query_box = query_box

    self.tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)
    self.tokenizer_cache = {}

    if self.image_seperator:
      self.tokenizer.add_tokens([self.IMAGE_SEP])
      image_sep_id = self.tokenizer.convert_tokens_to_ids([self.IMAGE_SEP])
      assert len(image_sep_id) == 1
      self._image_sep_id = image_sep_id[0]
    else:
      self._image_sep_id = None

    self.image_joiner = image_joiner

    self.preprocesser = Gpv1Preprocessor()
    self.preprocesser.init(self._pre_tokenize)

    self.model = None
    self.obj_cls = None

    # Prediction arguements
    self.nms = nms
    self.beam_search_spec = None
    self.register_buffer("mask", None)

  def _pre_tokenize(self, text: str) -> Union[str, np.ndarray]:
    if self.all_lower_case:
      text = text.lower()
    if self.pre_tokenize:
      return np.array(encode_with_cache(text, self.tokenizer, self.tokenizer_cache))
    else:
      return text

  def initialize(self):
    if self.initialize_t5:
      logging.info(f"Loading pre-trained LM {self.t5_model_name}")
      self.model: OurT5ForConditionalGeneration = OurT5ForConditionalGeneration.from_pretrained(self.t5_model_name)
    else:
      config = AutoConfig.from_pretrained(self.t5_model_name)
      self.model = OurT5ForConditionalGeneration(config)

    self._init_non_pretrained()

    if self.initialize_joiner:
      if self.initialize_joiner == "coco":
        words = ID_TO_COCO_CATEGORY.values()
      else:
        raise NotImplementedError()
      logging.info("Initializing joiner bias with mean embeddings")
      if isinstance(self.image_joiner, Linear):
        all_tokens = set()
        for cat in words:
          all_tokens.update(encode_with_cache(cat, self.tokenizer, self.tokenizer_cache))
        all_tokens = torch.as_tensor(list(all_tokens), dtype=torch.long)
        self.image_joiner.bias.data[:] = self.model.shared(all_tokens).mean(0)
      else:
        raise NotImplementedError()

  def _init_non_pretrained(self):
    t5_dim = self.model.config.d_model
    n_heads = self.model.config.num_heads

    if self.image_relevance is None:
      self.obj_cls = nn.Linear(t5_dim, 2)

    if self.query_box is not None:
      self.query_embedding = nn.Parameter(torch.zeros(t5_dim).uniform_(-0.05, 0.05))

    if self.image_positional_bias == "learned":
      self.learned_image_text_bias = nn.Parameter(torch.zeros(n_heads,))
      self.learned_image_image_bias = nn.Parameter(torch.zeros(n_heads,))
      self.learned_text_image_bias = nn.Parameter(torch.zeros(n_heads,))
    else:
      self.learned_image_text_bias = None
      self.learned_image_image_bias = None
      self.learned_text_image_bias = None

    if self.freeze_embed:
      self.model.get_input_embeddings().weight.requires_grad = False
      self.model.get_output_embeddings().weight.requires_grad = False

  def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    if "mask" in state_dict:
      # In case it was built during `set_prediction_args` and accidentally saved
      del state_dict["mask"]

    if self.model is None:
      config = AutoConfig.from_pretrained(self.t5_model_name)
      self.model = OurT5ForConditionalGeneration(config)

    self._init_non_pretrained()

    if any(k.startswith("detr_joiner.") for k in state_dict):
      _fix_old_state_dict(state_dict)

    super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  def preprocess_example_train(self, example):
    return self.preprocesser.preprocess_example(example, True, self.query_box is not None)

  def preprocess_example(self, example, is_train=False):
    return self.preprocesser.preprocess_example(example, is_train, self.query_box is not None)

  def get_collate(self, is_train=False):
    n_pos = self.model.config.n_positions
    return CollateWithTokenizer(
      self.tokenizer, self.image_feature_extractor.get_collate(is_train),
      n_pos, n_pos, self.pre_tokenize)

  def _get_encoder_pos_bias(self, seq_len, n_image):
    input_pos_bias = None
    if self.image_positional_bias in {"zero", "learned"}:
      first_self_atten = self.model.encoder.block[0].layer[0].SelfAttention
      if self.image_positional_bias == "zero":
        input_pos_bias = first_self_atten.compute_bias(seq_len, seq_len)
        input_pos_bias[:, :, :n_image, :] = 0
        input_pos_bias[:, :, :n_image] = 0
      elif self.image_positional_bias.startswith("learned"):
        n_heads = self.model.config.num_heads
        n_text = seq_len - n_image
        i_t_bias = self.learned_image_text_bias.view(1, n_heads, 1, 1).repeat(1, 1, n_image, n_text)
        t_i_bias = self.learned_text_image_bias.view(1, n_heads, 1, 1).repeat(1, 1, n_text, n_image)
        i_i_bias = self.learned_image_image_bias.view(1, n_heads, 1, 1).repeat(1, 1, n_image, n_image)
        t_t_bias = first_self_atten.compute_bias(n_text, n_text)
        input_pos_bias = torch.cat([
          torch.cat([i_i_bias, i_t_bias], 3),
          torch.cat([t_i_bias, t_t_bias], 3)
        ], 2)
      else:
        raise NotImplementedError()
    elif self.image_positional_bias is not None:
      raise NotImplementedError(self.image_positional_bias)
    return input_pos_bias

  def _encode(self, image_inputs, input_ids, input_mask):
    image: ImageRegionFeatures = self.image_feature_extractor(**image_inputs)

    device = image.features.device

    if isinstance(self.image_joiner, Linear):
      detr_embed = self.image_joiner(image.features)
    else:
      detr_embed = self.image_joiner(image.features, self.model.shared)

    if self.query_box == "always":
      # Adds a query embeddings to the query boxes
      if image.n_boxes is not None:
        batch_ixs = torch.arange(len(detr_embed), device=device, dtype=torch.long)
        end_ixs = image.n_boxes - 1
        detr_embed[batch_ixs, end_ixs] += self.query_embedding
      else:
        detr_embed[:, -1] += self.query_embedding
    elif self.query_box is None:
      pass
    else:
      raise NotImplementedError(self.query_box)

    if self.image_seperator:
      input_ids = torch.cat([
        input_ids,
        torch.full_like(input_ids[:, :1], self._image_sep_id),
      ], 1)
      input_mask = torch.cat([input_mask[:, :1], input_mask], 1)

    query_embed = self.model.shared(input_ids)

    query_embed, input_mask = our_utils.concat_masked_sequences(
      detr_embed, image.n_boxes,
      query_embed, input_mask
    )

    if self.query_box == "always":
      # Remove the query box so our output only contains the object boxes
      if image.n_boxes is not None:
        image.n_boxes = image.n_boxes - 1
      image.boxes = image.boxes[:, :-1]
      image.features = image.features[:, :-1]
      image.objectness = image.objectness[:, :-1]
    elif self.query_box is None:
      pass
    else:
      raise NotImplementedError(self.query_box)

    input_pos_bias = self._get_encoder_pos_bias(input_ids.size(1), image.features.size(1))

    encoder_outputs = self.model.encoder(
      inputs_embeds=query_embed,
      attention_mask=input_mask,
      encoder_positional_bias=input_pos_bias,
      return_dict=True
    )
    return encoder_outputs, input_mask, image

  def _image_rel(self, encoder, image: ImageRegionFeatures):
    if self.image_relevance is None:
      n_images = image.boxes.size(1)
      return self.obj_cls(encoder[:, :n_images])
    else:
      rel = self.image_relevance(encoder, image.objectness, image.boxes)
      if len(rel.size()) == 2:
        return torch.stack([rel, torch.zeros_like(rel)], -1)
      else:
        return rel

  def forward(self, image_inputs, input_ids, input_mask, labels=None,
              box_targets=None, tasks=None, label_masks=None) -> Tuple[torch.Tensor, Dict[str, float]]:

    encoder_outputs, input_mask, image_features = self._encode(image_inputs, input_ids, input_mask)

    t5_out = self.model(
      encoder_outputs=encoder_outputs,
      attention_mask=input_mask,
      labels=labels,
      return_dict=True
    )

    if not self.predict_trailing_pad_tokens:
      # -100 marks a label as not a target
      labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

    rel = self._image_rel(t5_out.encoder_last_hidden_state, image_features)

    boxes = image_features.boxes
    n_boxes = image_features.n_boxes

    loss, monitor = self.loss(t5_out.logits, labels, boxes, rel, n_boxes, box_targets, tasks)
    return loss, monitor

  def set_prediction_args(
      self,
      beam_search_spec: BeamSearchSpec=None,
      answer_options=None, mask=None, nms=None
  ):
    self.beam_search_spec = beam_search_spec
    if nms is not None:
      self.nms = nms

    voc_len = self.model.config.vocab_size
    device = our_utils.get_model_device(self)

    if mask is not None:
      words = mask.get_target_words()
      tensor_mask = np.zeros([voc_len], dtype=np.bool)
      for word in words:
        tensor_mask[encode_with_cache(word, self.tokenizer, self.tokenizer_cache)] = True
      for word in ID_TO_COCO_CATEGORY.values():
        if word not in words:
          tensor_mask[encode_with_cache(word, self.tokenizer, self.tokenizer_cache)] = False
      tensor_mask[self.tokenizer.eos_token_id] = mask.target_eos()
      tensor_mask = torch.as_tensor(tensor_mask, device=device)
      self.register_buffer("mask", tensor_mask.float() * mask.val, persistent=False)
    else:
      self.register_buffer("mask", None, persistent=False)

    if answer_options is not None:
      eos = self.tokenizer.eos_token_id
      tokenized = [encode_with_cache(x, self.tokenizer, self.tokenizer_cache) + [eos]
                   for x in answer_options]
      answer_mask = np.zeros((max(len(x) for x in tokenized), voc_len), dtype=np.bool)
      for tok in tokenized:
        answer_mask[np.arange(len(tok)), tok] = True
      answer_mask = torch.as_tensor(answer_mask, device=device).float()
      # Word pieces that can never be part of an answer option get a large negative weight
      answer_mask = (1 - answer_mask) * -1e9
      if self.mask is not None:
        answer_mask = answer_mask + self.mask.unsqueeze(0)
      self.register_buffer("mask", answer_mask)

  def predict(
      self, image_inputs, input_ids, input_mask, labels=None,
      box_targets=None, tasks=None,
  ):
    task = tasks[0]
    if not all(x == task for x in tasks):
      raise NotImplementedError("Predict requires all examples have the same batch")

    encoder_outputs, input_mask, image_features = self._encode(image_inputs, input_ids, input_mask)
    rel = self._image_rel(encoder_outputs.last_hidden_state, image_features)
    rel = rel.softmax(-1)[:, :, 0]

    if self.beam_search_spec is not None:
      if self.mask is None:
        post_process = None
      else:
        def post_process(logits, _, time_step):
          if len(self.mask.size()) == 1:
            return F.log_softmax(logits + self.mask, -1)
          else:
            return F.log_softmax(logits + self.mask[time_step], -1)
      bs = self.beam_search_spec.build(self.tokenizer.eos_token_id)
      decode_init = t5_initialize_decoding(
        self.tokenizer, self.model, encoder_outputs[0], input_mask, post_process)
      input_ids, logprobs = bs.search(*decode_init)
      input_ids = input_ids.detach().cpu()

      out_text = []
      for batch in range(len(input_ids)):
        text = [self.post_process_generation(x) for x in input_ids[batch]]
        out_text.append(text)
    else:
      out_text, logprobs = None, None

    if self.nms is not None:
      for i in range(image_features.boxes.size(0)):
        boxes = image_features.boxes[i]
        boxes = torchvision.ops.box_convert(boxes, "cxcywh", "xyxy")
        keep = torchvision.ops.nms(boxes, rel[i], self.nms)
        to_subtract = torch.full_like(rel[i], 10000)
        to_subtract[keep] = 0
        rel[i] -= to_subtract

    return build_per_example_output(
      out_text, logprobs, image_features.boxes, rel, image_features.n_boxes)

  def post_process_generation(self, generated_ids):
    return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
