from copy import Error
import logging
from typing import Union, Tuple, Dict, Optional, Callable, List
import os
import torch
import torchvision.ops
from allennlp.common import Params
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
import pdb
from exp.ours.data.dataset import Task
from exp.ours.data.gpv import COCO_CATEGORIES
from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.webqa_templates import WebQaQueryGenerator, DefaultWebQueryGenerator
from exp.ours.models.model import GPVModel, build_per_example_output
from exp.ours.models.gpv1_preprocessing import Gpv1Preprocessor
from exp.ours.image_featurizer.image_featurizer import ImageFeatureExtractor, ImageRegionFeatures, \
  ImageCollater
from exp.ours.models.layers import Layer, Linear
from exp.ours.models.losses import GPVLoss, BasicGPVLoss, GpvBatchPrediction, GpvBatchLabels
import numpy as np

from exp.ours.models.model_utils import CollateWithTokenizer
from exp.ours.models.t5_custom import OurT5ForConditionalGeneration
from exp.ours.train.runner import BeamSearchSpec
from exp.ours.util import our_utils
from exp.ours.util.nlp_utils import encode_with_cache, t5_initialize_decoding
from torch import nn
from torch.nn import functional as F
from exp.ours.util.to_params import to_params
import utils.io as io 


@dataclass
class CollateLocalizationLabels:
  tokenizer: PreTrainedTokenizer

  def collate(self, batch: List[GPVExample], out):
    # Get the tokenized labels for each detection example
    
    per_box_labels = []
    for ex in batch:
      #print(ex.task==Task.IMAGECONTRAST,'task')
      if ex.task == Task.DETECTION or ex.task == Task.IMAGECONTRAST or ex.task == Task.MIL or ex.task == Task.SYNONYM or ex.task == Task.TEXTCONTRAST:
        assert ex.relevance_query is not None
        #print('examples')
        per_box_labels.append(ex.relevance_query)
      else:
        per_box_labels.append(self.tokenizer.pad_token)
    #print(per_box_labels,len(per_box_labels),'per box labels')
    labels = self.tokenizer(
      per_box_labels, return_tensors='pt', padding=True, truncation=True)
    return dict(relevance_queries=labels["input_ids"].view(len(batch), -1))


@GPVModel.register("t5-gpv-per-box")
class T5GpvPerBox(GPVModel):
  IMAGE_SEP = "[IMAGE]"

  @classmethod
  def from_params(
    cls,
    params: Params,
    **kwargs,
  ):
    # Handle various backwards compatibility issues
    if "box_aux_loss" in params:
      assert params.pop("box_aux_loss") is None
    if "nms" in params and params["nms"] == 0.0:
      params["nms"] = None
    if "webqa_templates" not in params:
      params["webqa_templates"] = None
    if "relevance_embedding" not in params and "relevance_conditioning" in params:
      params["relevance_embedding"] = params.pop("relevance_conditioning")
    if "relevance_conditioning" in params:
      # No longer supported
      assert params.pop("relevance_conditioning") is None

    webqa_templates = params["webqa_templates"]
    if webqa_templates is not None:
      if "oversample_questions" in webqa_templates:
        if "type" not in webqa_templates:
          webqa_templates["type"] = "templated-v1"
      elif "type" not in webqa_templates:
        params["webqa_templates"] = None

    if "image_relevance" in params:
      del params["image_relevance"]

    return super().from_params(params, **kwargs)

  def __init__(
      self,
      t5_model_name: str,
      loss: GPVLoss,
      image_feature_extractor: ImageFeatureExtractor,
      image_joiner: Layer,
      embed_objectness_score=False,
      initialize_t5=True, pre_tokenize=True,
      query_box=None,
      predict_trailing_pad_tokens=False,
      image_positional_bias="zero",
      image_seperator=False,
      initialize_joiner="coco",
      nms: float=None,
      all_lower_case=False,
      webqa_templates: Optional[WebQaQueryGenerator]=None,
      initialize_from=None,
      loc_relevance_query="category",
      convert_to_relevance="sigmoid-logits",
      cls_from_query_w: float=0.0,
      use_image_sep: bool=False,
      combine_with_objectness="multiply",
      contrast_query=None,
      box_context="none",
  ):
    super().__init__()
    if cls_from_query_w != 0:
      raise NotImplementedError()

    if nms == 0.0:
      # While this is technically possible, it doesn't really make sense to do nms with a 0.0
      # threshold and some old code conflated nms=0.0 with nms=None so we disallow nms=0.0 here
      # so we can treat 0.0 as None when loading this from a parameter file.
      raise ValueError("Need nms > 0.0")
    self.box_context = box_context
    self.all_lower_case = all_lower_case
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
    if query_box == "none":
      query_box = None
    self.query_box = query_box
    self.initialize_from = initialize_from
    self.embed_objectness_score = embed_objectness_score
    self.cls_from_query_w = cls_from_query_w
    self.use_image_sep = use_image_sep
    self.combine_with_objectness = combine_with_objectness
    self.contrast_query = contrast_query
    self.from_query_box = False
    self.loc_relevance_query = loc_relevance_query
    self.convert_to_relevance = convert_to_relevance

    if webqa_templates is None:
      webqa_templates = DefaultWebQueryGenerator()
    self.webqa_templates = webqa_templates

    self.tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)

    if self.contrast_query is not None:
      self.register_buffer("contrast_query_tok", self.tokenizer(
        self.contrast_query, return_tensors='pt', padding=True, truncation=True)["input_ids"][0])

    # Speed up tokenization by caching per-token tokenization
    self.tokenizer_cache = {}

    # Store text->tokens data, mainly used to reduce memory usage so we don't
    # build new objects for duplicate queries/answers
    self.full_text_cache = {}

    if self.image_seperator:
      self.tokenizer.add_tokens([self.IMAGE_SEP])
      image_sep_id = self.tokenizer.convert_tokens_to_ids([self.IMAGE_SEP])
      assert len(image_sep_id) == 1
      self._image_sep_id = image_sep_id[0]
    else:
      self._image_sep_id = None

    self.image_joiner = image_joiner

    self.preprocesser = Gpv1Preprocessor(self.webqa_templates, relevance_query=self.loc_relevance_query)
    self.preprocesser.init(self._pre_tokenize)

    self.model = None

    # Prediction arguements
    self.rerank_answer_options = None
    self.nms = nms
    self.beam_search_spec = None
    self.register_buffer("mask", None)

  def _pre_tokenize(self, text: str) -> Union[str, np.ndarray]:
    if self.all_lower_case:
      text = text.lower()
    if not self.pre_tokenize:
      return text
    if text in self.full_text_cache:
      return self.full_text_cache[text]
    else:
      out = np.array(encode_with_cache(text, self.tokenizer, self.tokenizer_cache))
      self.full_text_cache[text] = out
      return out

  def initialize(self, load_params=True):
    if self.initialize_from is not None:
      logging.info(f"Initializing model from {self.initialize_from}")
      state_dict = torch.load(self.initialize_from)
      if state_dict["image_joiner.weight"].size() != self.image_joiner.weight.size():
        state_dict["image_joiner.weight"] = F.pad(state_dict["image_joiner.weight"], [0, 5, 0, 0],)
      missing_key, unexpected_key = self.load_state_dict(state_dict, strict=False)
      logging.info(f"Missing keys {missing_key}")
      # self.relevance_rescale.bias.data[:] = 0
      # self.relevance_rescale.weight.data[:] = 0
      # self.relevance_rescale.weight.data[0, 0] = 1.0
      return

    if self.initialize_t5:
      logging.info(f"Loading pre-trained LM {self.t5_model_name}")
      self.model: OurT5ForConditionalGeneration = OurT5ForConditionalGeneration.from_pretrained(self.t5_model_name)
    else:
      config = AutoConfig.from_pretrained(self.t5_model_name)
      self.model = OurT5ForConditionalGeneration(config)

    self._init_non_pretrained()

    if self.initialize_joiner:
      if self.initialize_joiner == "coco":
        words = COCO_CATEGORIES
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

    self.relevance_rescale = nn.Linear(2, 2)
    if self.combine_with_objectness == "multiply-rescale":
      self.objectness_factor = nn.Linear(t5_dim, 1)

    # Initialize to directly use log_probability as relevance
    self.relevance_rescale.bias.data[:] = 0
    self.relevance_rescale.weight.data[:] = 0
    self.relevance_rescale.weight.data[0, 0] = 1.0

    if self.embed_objectness_score:
      self.objectness_embed = nn.Parameter(torch.zeros(t5_dim))
    else:
      self.objectness_embed = None

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

  def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    if "mask" in state_dict:
      # In case it was built during `set_prediction_args` and accidentally saved
      del state_dict["mask"]

    state_dict = dict(state_dict)
    for k in list(state_dict):
      if k.startswith("image_relevance."):
        del state_dict[k]

    if self.model is None:
      config = AutoConfig.from_pretrained(self.t5_model_name)
      self.model = OurT5ForConditionalGeneration(config)

    self._init_non_pretrained()

    super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  def preprocess_example_train(self, example):
    return self.preprocess_example(example, is_train=True)

  def preprocess_example(self, example, is_train=False):
    return self.preprocesser.preprocess_example(
      example, is_train, self.query_box is not None)

  def get_collate(self, is_train=False):
    n_pos = self.model.config.n_positions
    return CollateWithTokenizer(
      self.tokenizer, self.image_feature_extractor.get_collate(is_train),
      n_pos, n_pos, self.pre_tokenize,
      CollateLocalizationLabels(self.tokenizer))

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

    if self.embed_objectness_score:
      detr_embed += torch.exp(image.objectness).unsqueeze(2) * self.objectness_embed.unsqueeze(0).unsqueeze(0)

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
        torch.full_like(input_ids[:, :1], self._image_sep_id),
        input_ids,
      ], 1)
      input_mask = torch.cat([input_mask[:, :1], input_mask], 1)

    query_embed = self.model.shared(input_ids)

    query_embed, input_mask = our_utils.concat_masked_sequences(
      detr_embed, image.n_boxes,
      query_embed, input_mask
    )

    input_pos_bias = self._get_encoder_pos_bias(input_ids.size(1), image.features.size(1))

    encoder_outputs = self.model.encoder(
      inputs_embeds=query_embed,
      attention_mask=input_mask,
      encoder_positional_bias=input_pos_bias,
      return_dict=True
    )
    return encoder_outputs, input_mask, image

  def _get_rel_logprob(self, objectness):
    if len(objectness.size()) == 2:
      objectness = our_utils.log_prob_to_logits(objectness)
    else:
      if objectness.size(2) > 2:
        non_object_lp = F.log_softmax(objectness, -1)[:, :, -1]
        object_lp = torch.log1p(-torch.exp(non_object_lp))
        objectness = torch.stack([object_lp, non_object_lp], -1)
      else:
        assert objectness.size(2) == 2
        objectness = F.log_softmax(objectness, -1)
    return objectness

  def _image_rel(self, image_features: ImageRegionFeatures, box_rel, contextual_embeds):
    """Converts box-scores and image_features to relevance scores"""
    #print(len(box_rel.size()),'box rel size')
    if self.convert_to_relevance == "raw":
      box_scores = self.relevance_rescale(box_rel)

    elif self.convert_to_relevance == "sigmoid-logits":
      if len(box_rel.size()) == 2:
        assert torch.all(torch.isfinite(box_rel))
        box_rel = our_utils.log_prob_to_logits(box_rel)
        assert torch.all(torch.isfinite(box_rel))
      else:
        box_rel = torch.log_softmax(box_rel, -1)

      # Re-calibrate now [batch, n_boxes, 2] logits
      box_scores = self.relevance_rescale(box_rel)
      #print(box_scores,'box scores')
    else:
      raise NotImplementedError(self.convert_to_relevance)

    objectness = self._get_rel_logprob(image_features.objectness)

    if self.combine_with_objectness == "none":
      pass
    elif self.combine_with_objectness == "multiply":
      box_scores = F.log_softmax(box_scores, -1)
      box_scores = objectness + box_scores
    elif self.combine_with_objectness == "multiply-rescale":
      box_scores = F.log_softmax(box_scores, -1)
      factor = self.objectness_factor(contextual_embeds[:, :box_rel.size(1)])
      box_scores = objectness*factor + box_scores
    else:
      raise ValueError()
    #print(box_scores,'box scores')
    return box_scores

  def compute_per_box_score(
      self, tasks, contextual_emb, n_boxes, relevance_query, input_mask,
      include_query=False
  ):
    """
    @returns [batch, n_boxes], or [batch, n_boxes, 2] if there is a contrastive query,
    of per-box generation log-probabilities
    """
    device = contextual_emb.device
    #print('made it to line 431')
    if self.use_image_sep:
      # Make sure to include the image seperator
      raise ValueError()

    if not self.predict_trailing_pad_tokens:
      # -100 marks a label as not a target
      relevance_query = relevance_query.masked_fill(
        relevance_query == self.tokenizer.pad_token_id, -100)
    #print('made it to line 440')
    if not include_query:
      if self.query_box is not None:
        assert self.query_box == "always"
        n_boxes = n_boxes - 1
    #print('made it to line 445')
    per_box_inputs_lst = []
    per_box_outputs_lst = []
    ixs = []
    for i, task in enumerate(tasks):
      if not (task == Task.DETECTION or task == Task.IMAGECONTRAST or task == Task.MIL or task==Task.SYNONYM or task==Task.TEXTCONTRAST):
        continue
      #print('made it to line 452')
      ixs.append(i)
      if self.box_context == "none" or self.box_context is None:
        context = None
      elif self.box_context == "query":
        assert self.query_box == "always"
        #print('made it to line 458')
        query_start = n_boxes[i]
        context = contextual_emb[i, query_start:input_mask[i].sum()]
      else:
        if self.box_context == "query_end":
          ix = input_mask[i].sum() - 1
        else:
          raise ValueError()

        context = contextual_emb[i, ix].unsqueeze(0)
      #print('made it to line 468')
      box_emb = contextual_emb[i, :n_boxes[i]].unsqueeze(1)

      if context is not None:
        context = context.unsqueeze(0).repeat(box_emb.size(0), 1, 1)
        box_emb = torch.cat([box_emb, context], 1)
      #print('made it to line 474')
      per_box_inputs_lst.append(box_emb)
      per_box_outputs_lst.append(relevance_query[i].unsqueeze(0).repeat(box_emb.size(0), 1))

    # Build a tensor for the [batch, n_boxes] sparse representation we will fill out
    if self.contrast_query is not None:
      batched_rel_scores = torch.full(
        (n_boxes.size(0), n_boxes.max(), 2),
        -10000,
        device=device, dtype=torch.float
      )
    else:
      batched_rel_scores = torch.full(
        (n_boxes.size(0), n_boxes.max()),
        -10000,
        device=device, dtype=torch.float
      )
      #print(batched_rel_scores.size(),'batched rel scores')
    #print('made it to line 492')
    if len(ixs) == 0:
      #print('hi')
      return batched_rel_scores
    #print('made it to line 496')
    per_box_inputs, per_box_mask = our_utils.stack_and_pad_blocks(per_box_inputs_lst)
    per_box_outputs = torch.cat(per_box_outputs_lst, 0)
    #print('made it to line 499',  per_box_inputs.size(0) == per_box_outputs.size(0))
    assert per_box_inputs.size(0) == per_box_outputs.size(0)
    total_boxes = per_box_inputs.size(0)

    t5_out = self.model(
      encoder_outputs=(per_box_inputs,),
      attention_mask=per_box_mask,
      labels=per_box_outputs,
      return_dict=True,
    )
    #print('made it to line 510')
    #print(per_box_outputs.view(-1),'per box outputs')
    
    dim = t5_out.logits.size(-1)
    per_label_score = F.cross_entropy(
      t5_out.logits.view(-1, dim), per_box_outputs.view(-1), reduction="none")
    #print('made it to line 516')
    per_label_score = -per_label_score.view(total_boxes, -1).sum(1)
    #print(per_label_score,'per label score')
    #print('made it to line 519')
    if self.contrast_query is not None:
      c_labels = self.contrast_query_tok.unsqueeze(0).repeat(total_boxes, 1)
      #print('made it to line 522')
      contrast_query_out = self.model(
        encoder_outputs=(per_box_inputs,),
        attention_mask=per_box_mask,
        labels=c_labels,
        return_dict=True,
      )
      #print('made it to line 525')
      c_per_label_score = F.cross_entropy(
        contrast_query_out.logits.view(-1, dim), c_labels.view(-1), reduction="none")
      #$print('made it to line 532')
      c_per_label_score = -c_per_label_score.view(total_boxes, -1).sum(1)
      #print('made it to line 534')
      per_label_score = torch.stack([per_label_score, c_per_label_score], -1)
      #print(per_label_score.size(),'per label score')

    on = 0
    for i in ixs:
      n = n_boxes[i]
      batched_rel_scores[i, :n] = per_label_score[on:on+n]
      on = on + n
    assert on == len(per_label_score)
    #print(batched_rel_scores.size(),'final batched rel scores ')
    return batched_rel_scores

  def forward(self, image_inputs, input_ids, input_mask, labels: GpvBatchLabels,
              relevance_queries) -> Tuple[torch.Tensor, Dict[str, float]]:
    #print(input_ids,'input ids')
    #print('made it here')
    encoder_outputs, input_mask, image_features = self._encode(image_inputs, input_ids, input_mask)
    #print('Encoded the image')
    #print(len(input_ids),'input ids')
    # per_box_scores = self.compute_per_box_score(
    #   labels.tasks, encoder_outputs.last_hidden_state,
    #   image_features.get_n_boxes(), relevance_queries, input_mask, True)
    per_box_scores = self.compute_per_box_score(
        labels.tasks, encoder_outputs.last_hidden_state,
        image_features.get_n_boxes(), relevance_queries, input_mask, True)
    # try:
     
    #   #print(per_box_scores.size(),'perbox scores')
    #   #print('Computed box scores')
    # except RuntimeError:
    #   print(encoder_outputs.last_hidden_state,input_mask.size(),image_features.get_n_boxes(),relevance_queries,input_mask)
    
    
    #print(per_box_scores.size(),'per box scores size')
    rel = self._image_rel(image_features, per_box_scores, encoder_outputs.last_hidden_state)
    # try:
      
    # except UnboundLocalError:
    #   raise UnboundLocalError
    #print(rel.size(),'relevance scores')
    #print('Computed relevance')
    t5_out = self.model(
      encoder_outputs=encoder_outputs,
      attention_mask=input_mask,
      labels=labels.text_labels,
      return_dict=True,
    )
  
    #print(labels.text_labels,'text labels')
    #print('Computed out')
    if not self.predict_trailing_pad_tokens:
      # -100 marks a label as not a target
      labels.text_labels = labels.text_labels.masked_fill(
        labels.text_labels == self.tokenizer.pad_token_id, -100)

    boxes = image_features.boxes
    n_boxes = image_features.n_boxes
    #print(rel.size(),'relevance size')
    #print(len(t5_out.logits),'t5 logits')
    #print(boxes,rel,'boxes rel')
    
    batch_pred = GpvBatchPrediction(t5_out.logits, boxes, rel, n_boxes)
    print(batch_pred.logits.size(),'batch pred logits')
    print(batch_pred.pred_boxes.size(),'pred boxe sizez')
    print(batch_pred.pred_rel.size(),'pred rel size')
    print(len(labels.image_ids),'images ')
    json_dump = {'pred_boxes':batch_pred.pred_boxes.detach().cpu().numpy(),'rel':batch_pred.pred_rel.cpu().detach().numpy(),
    'images':labels.image_ids,'queries':labels.queries}
    if not os.path.exists('/home/michal/gpv_michal/syn_values.json'):
      io.dump_json_object(json_dump,'/home/michal/gpv_michal/syn_values.json')
    

    #print('Computed batch pred')
    #print(rel,'batch pred')
    #print(batch_pred,'batch pred')
    loss, monitor = self.loss(batch_pred, labels)
    #print('Computed loss')
    return loss, monitor,json_dump


  def set_prediction_args(
      self,
      beam_search_spec: BeamSearchSpec=None,
      answer_options=None, mask=None, nms=None,
      rerank_answer_options=False,
  ):
    if rerank_answer_options:
      if answer_options is None:
        raise ValueError("No answer options to re-rank!")
      self.rerank_answer_options = answer_options
    else:
      self.rerank_answer_options = None

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
      for word in COCO_CATEGORIES:
        if word not in words:
          tensor_mask[encode_with_cache(word, self.tokenizer, self.tokenizer_cache)] = False
      tensor_mask[self.tokenizer.eos_token_id] = mask.target_eos()
      tensor_mask = torch.as_tensor(tensor_mask, device=device)
      self.register_buffer("mask", tensor_mask.float() * mask.val, persistent=False)
    else:
      self.register_buffer("mask", None, persistent=False)

    self.register_buffer("answer_ids", None)

    if answer_options is not None:
      if rerank_answer_options:
        if beam_search_spec:
          raise ValueError("No beam search if we just doing re-ranking")
        tokenized_answers = self.tokenizer(
          answer_options, return_tensors='pt', padding=True, max_length=self.model.config.n_positions)
        labels = tokenized_answers["input_ids"].to(device)
        if not self.predict_trailing_pad_tokens:
          # -100 marks a label as not a target
          labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        self.register_buffer("answer_ids", labels, persistent=False)
      else:
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
        self.register_buffer("mask", answer_mask, persistent=False)

  def predict(
      self, image_inputs, input_ids, input_mask, labels: GpvBatchLabels, relevance_queries=None,json_output=False):
    # Use no_grad just so clients don't have to remember to
    with torch.no_grad():
      return self._predict(image_inputs, input_ids, input_mask, labels, relevance_queries,json_output)

  def _predict(
      self, image_inputs, input_ids, input_mask, labels: GpvBatchLabels,
      relevance_queries=None,json_output=False
  ):
    task = labels.tasks[0]
    if not all(x == task for x in labels.tasks):
      raise NotImplementedError("Predict requires all examples have the same batch")
    #print(task,'TASK')
    encoder_outputs, input_mask, image_features = self._encode(image_inputs, input_ids, input_mask)
    if task == (Task.DETECTION or Task.IMAGECONTRAST or Task.TEXTCONTRAST or Task.SYNONYM or Task.MIL):
      #print('hi using this')
      per_box_scores = self.compute_per_box_score(
        labels.tasks, encoder_outputs.last_hidden_state,
        image_features.get_n_boxes(), relevance_queries, input_mask, True)
      #print(per_box_scores,'per box scores for prediction')
      #print(per_box_scores.size(),'per box scores size for prediction')
      rel = self._image_rel(image_features, per_box_scores, encoder_outputs.last_hidden_state)
      #print(rel.size(),'final image rel for prediction')
      #print(rel,'rel')
      rel = rel.softmax(-1)[:, :, 0]
    else:
      if len(image_features.objectness.size()) == 3:
        objectness = self._get_rel_logprob(image_features.objectness)
        rel = F.softmax(objectness, -1)[:, :, 0]
      else:
        rel = torch.exp(image_features.objectness)

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

    elif self.rerank_answer_options:
      n_answers = len(self.answer_ids)
      n_queries, enc_len, dim = encoder_outputs.last_hidden_state.size()
      labels = self.answer_ids.unsqueeze(0).repeat(n_queries, 1, 1).view(n_queries*n_answers, -1)
      input_mask = input_mask.unsqueeze(1).repeat(1, n_answers, 1).view(n_queries*n_answers, -1)
      encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(1)\
        .repeat(1, n_answers, 1, 1).view(n_queries*n_answers, enc_len, dim)
      t5_out = self.model(
        encoder_outputs=encoder_outputs,
        attention_mask=input_mask,
        labels=labels,
        return_dict=True,
      )
      dec_len = labels.size(1)
      if self.mask is not None:
        t5_out.logits = F.log_softmax(t5_out.logits, -1) + self.mask
      per_answer_loss = F.cross_entropy(
        t5_out.logits.view(n_queries*n_answers*dec_len, -1),
        labels.view(n_queries*n_answers*dec_len), reduction="none"
      ).view(n_queries, n_answers, dec_len)
      per_answer_loss = per_answer_loss.sum(-1).cpu().numpy()
      answer_ranks = np.argsort(per_answer_loss, axis=1)
      out_text = []
      logprobs = []
      for batch in range(n_queries):
        out_text.append([self.rerank_answer_options[r] for r in answer_ranks[batch]])
        # Negative convert NLL to LL
        logprobs.append(-per_answer_loss[batch][answer_ranks[batch]])
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
    #print(rel.size(),'rel last line')
    return build_per_example_output(
      out_text, logprobs, image_features.boxes, rel, image_features.n_boxes)

  def post_process_generation(self, generated_ids):
    return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
