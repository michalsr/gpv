# TODO: Haven't finished adding web here: come back.

import copy
from inspect import signature

import torch
from typing import List, Dict, Optional
import numpy as np
from allennlp.common import FromParams, Params
from dataclasses import dataclass
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from omegaconf import OmegaConf
from torch import nn
from transformers import PreTrainedTokenizer, BertTokenizer

from exp.ours.boosting import MaskSpec
from exp.gpv.metrics import create_coco_vocab_mask
from exp.gpv.models.gpv import GPV
from exp.ours import file_paths
from exp.ours.data.dataset import *
from exp.ours.image_featurizer.image_featurizer import ImageFeatureExtractor, ImageCollater
from exp.ours.train.runner import BeamSearchSpec
from exp.ours.util import our_utils, py_utils

from exp.ours.data.gpv_example import GPVExample
from exp.ours.models.model import GPVModel
from exp.ours.util.to_params import to_params_any


def gpv1_convert(example, train):
  if isinstance(example, VqaExample):
    return [GPVExample(
      example.gpv_id,
      Task.VQA,
      example.image_id,
      example.question,
      target_boxes=None,
      target_answer=example.meta["gpv1-answer"],
      meta=example.meta
    )]
  elif isinstance(example, ClsExample):
    return [GPVExample(
      example.gpv_id,
      Task.CLS,
      example.image_id,
      example.meta["gpv1-query"],
      crop=example.crop,
      target_boxes=None,
      target_answer=example.category,
      meta=example.meta
    )]
  elif isinstance(example, LocalizationExample):
    return [GPVExample(
      example.gpv_id,
      Task.DETECTION,
      example.image_id,
      example.meta["gpv1-query"],
      example.bboxes,
      target_answer=None,
      meta=example.meta
    )]
  elif isinstance(example, CaptioningExample):
    if train:
      out = []
      for cap in example.captions:
        out.append(GPVExample(
          cap.gpv_id,
          Task.CAPTIONING,
          example.image_id,
          cap.meta["gpv1-query"],
          target_boxes=None,
          target_answer=cap.caption,
          meta=cap.meta
        ))
      return out
    else:
      return [GPVExample(
        example.gpv_id,
        Task.CAPTIONING,
        example.image_id,
        [cap.meta["gpv1-query"] for cap in example.captions],
        target_boxes=None,
        target_answer=[x.caption for x in example.captions],
        meta=example.meta
      )]
  else:
    raise ValueError(type(example))


def truncate_beam(beam):
  for i, w in enumerate(beam):
    if w == "__stop__" or w == "__pad__":
      return beam[:i]
  return beam


FULLNAMES = {
  Task.CAPTIONING: "CocoCaptioning",
  Task.VQA: "CocoVqa",
  Task.CLS: "CocoClassification",
  Task.DETECTION: "CocoDetection",
  Task.WEBQA: "WebQa"
}


def full_task_name(task: Task):
  return FULLNAMES[task]


def _fix_old_state_dict(state_dict):
  for k in list(state_dict):
    if k.startswith("gpv.detr."):
      v = state_dict.pop(k)
      k = "image_feature_extractor." + k[len("gpv."):]
      state_dict[k] = v


def load_gpv1(model_config, checkpoint, loc):
  model = GPV1(model_config)
  loaded_dict = torch.load(checkpoint, map_location=loc)['model']
  state_dict = model.gpv.state_dict()
  for k, v in state_dict.items():
    state_dict[k] = loaded_dict[f'module.{k}']

  model.gpv.load_state_dict(state_dict)
  model.eval()
  model.to(loc)
  return model


@GPVModel.register("gpv1")
class GPV1(GPVModel, FromParams):

  @classmethod
  def from_params(
        cls, params, constructor_to_call=None,
        constructor_to_inspect=None, **extras,
    ):
    init_signature = signature(cls.__init__)
    param_names = [p for p in init_signature.parameters.keys() if p not in {"self", "gpv_cfg"}]
    args = {}
    for name in param_names:
      if name in params:
        args[name] = params.pop(name)

    if "freeze_detr" in params:
      if not params["init_detr"]:
        raise NotImplementedError()
      params["image_feature_extractor"] = {'type': 'detr', 'freeze': params.pop("freeze_detr")}

    cfg = OmegaConf.create(params.as_dict())
    # Set absolute paths
    cfg.vocab = file_paths.GPV1_VOC
    cfg.vocab_embed = file_paths.GPV1_VOC_EMBED
    args["gpv_cfg"] = cfg
    return super().from_params(
      Params(args), constructor_to_call, constructor_to_inspect)

  def to_params(self):
    # Delete absolute paths
    cfg = copy.deepcopy(self.gpv_cfg)
    for x in ["pretr_detr", "vocab", "vocab_embed"]:
      if x in cfg:
        del cfg[x]
    params = OmegaConf.to_container(cfg, resolve=True)
    init_signature = signature(self.__init__)
    param_names = [p for p in init_signature.parameters.keys() if p not in {"self", "gpv_cfg"}]
    for name in param_names:
      annotation = init_signature.parameters[name].annotation
      assert name not in params
      params[name] = to_params_any(getattr(self, name), annotation)
    return params

  def __init__(
      self, gpv_cfg: OmegaConf,
      image_feature_extractor: ImageFeatureExtractor=None,
      image_feature_dim=2304,
      unseen_train_mask=False,
      _model=None
  ):
    super().__init__()
    self._model = _model
    self.image_feature_dim = image_feature_dim
    if image_feature_extractor is None:
      self.image_feature_extractor = PretrainedDetrRIOExtractor()
    else:
      self.image_feature_extractor = image_feature_extractor
    if _model is not None:
      self.gpv = _model
    else:
      self.gpv = GPV(gpv_cfg, detr=False)
    self.unseen_train_mask = unseen_train_mask
    self.gpv_cfg = gpv_cfg
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if self.domain_embed_selector is not None:
      dim = gpv_cfg.hidden_dim
      cat_to_ix = {k: i for i, k in enumerate(sorted(ID_TO_COCO_CATEGORY.values()))}
      self.domain_embedder = nn.Linear(len(cat_to_ix), dim*2)
      # embedder = self.gpv.answer_input_embedings.embedding_layer
      # for cat, i in cat_to_ix.items():
      #   token_ids = [self.gpv.word_to_idx[x] for x in word_tokenize(cat)]
      #   with torch.no_grad():
      #     mean_embed = embedder(torch.as_tensor(token_ids)).mean(0)
      #   self.domain_embedder.weight.data[:mean_embed.size(0), i] = mean_embed / len(cat_to_ix)

    self.detok = TreebankWordDetokenizer()
    self.beam_size = 1

  def freeze_detr_params(self, requires_grad=False):
    for n, p in self.gpv.detr.named_parameters():
      if not n.startswith("class_embed."):
        p.requires_grad = requires_grad

  def initialize(self):
    pass

  def preprocess_example_train(self, example):
    return gpv1_convert(example, True)

  def preprocess_example(self, example):
    examples = gpv1_convert(example, False)
    assert len(examples) == 1
    return examples[0]

  def get_collate(self, is_train=False):
    return GPV1CollateFn(
      is_train,
      self.tokenizer,
      self.image_feature_extractor.get_collate(False),
      self.gpv.word_to_idx, self.gpv_cfg.max_text_len,
      self.gpv_cfg.answering_type, self.domain_embed_selector,
      self.unseen_train_mask
    )

  def forward(self, image_inputs, queries, answer_token_ids=None, targets=None, vocab_mask=None,
              task=None, domain_embed=None):
    if vocab_mask is not None:
      vocab_mask = vocab_mask.float()*(-10e20)

    if domain_embed is not None:
      domain_embed = torch.split(self.domain_embedder(domain_embed), [768, 768], -1)

    image_features = self.image_feature_extractor(**image_inputs)
    out = self.gpv(
      image_features, queries, answer_token_ids, targets,
      vocab_mask=vocab_mask,
      return_all_losses=True, domain_embed=domain_embed
    )
    if targets is None:
      return out
    loss, loss_dict = out
    monitor = {k: v.item() for k, v in loss_dict.items() if v is not None}

    return loss, monitor

  def _get_cls_mask(self):
    if not hasattr(self, "_classification_mask"):
      tokens, vocab_mask = create_coco_vocab_mask(self.gpv)
      device = our_utils.get_model_device(self)
      vocab_mask = torch.FloatTensor(vocab_mask).to(device)
      self.register_buffer("_classification_mask", vocab_mask)
    return self._classification_mask

  def predict(
      self,
      image_inputs, queries, answer_token_ids=None, targets=None, vocab_mask=None,
      domain_embed=None, allennlp_spec: BeamSearchSpec=None, predict_text=True,
      mask: Optional[MaskSpec]=None, use_mask=True
  ):
    if domain_embed is not None:
      domain_embed = torch.split(self.domain_embedder(domain_embed), [768, 768], -1)

    if not use_mask:
      vocab_mask = None

    imgs = self.image_feature_extractor(**image_inputs)

    if not predict_text:
      outputs = self.gpv(imgs, queries, answer_token_ids=None, vocab_mask=None)
      rel = outputs['pred_relevance_logits'].softmax(-1)[:, :, 0]
      boxes = outputs["pred_boxes"]
      return GPVOutput(boxes, rel, None, None)

    task = targets[0]["task"]
    if not all(x["task"] == task for x in targets):
      raise NotImplementedError("Predict requires all examples have the same batch")
    task = {v: k for k, v in FULLNAMES.items()}[task]

    if mask is not None:
      buffer_str = f"_{mask.get_name()}_{str(task)}"
      if not hasattr(self, buffer_str):
        words = mask.get_target_words(task)
        tokens = set(py_utils.flatten_list(word_tokenize(x) for x in words))
        voc_len = len(self.gpv.vocab)
        tensor_mask = np.zeros([voc_len], dtype=np.bool)
        for tok in tokens:
          tensor_mask[self.gpv.word_to_idx[tok]] = True
        tensor_mask = torch.as_tensor(tensor_mask, device=imgs.tensors.device)
        self.register_buffer(buffer_str, tensor_mask)

      tensor_mask = getattr(self, buffer_str)
      tensor_mask = tensor_mask.float() * mask.val
      if vocab_mask is not None:
        raise NotImplementedError()
    else:
      tensor_mask = None

    if allennlp_spec is not None:
      outputs, start_pred, initial_state, decode_fn = self.gpv.init_beam_search(
        imgs, queries, vocab_mask=tensor_mask, domain_embed=domain_embed)
      end_index = self.gpv.word_to_idx['__stop__']
      bs = allennlp_spec.build(end_index)
      input_ids, logprobs = bs.search(start_pred, initial_state, decode_fn)
      input_ids = input_ids.detach().cpu()

    else:
      outputs = self.gpv(imgs, queries)
      raise NotImplementedError()
    rel = outputs['pred_relevance_logits'].softmax(-1)[:, :, 0]
    boxes = outputs["pred_boxes"]
    out_text = []
    for batch in range(len(input_ids)):
      text = [self.post_process_generation(x) for x in input_ids[batch]]
      out_text.append(text)
    return GPVOutput(boxes, rel, out_text, logprobs)

  def post_process_generation(self, generated_ids):
    tokens = self.gpv.token_ids_to_words(generated_ids.unsqueeze(0))[0]
    tokens = truncate_beam(tokens)
    return self.detok.detokenize(tokens)

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
    if prefix + "_classification_mask" in state_dict:
      del state_dict[prefix + "_classification_mask"]

    if "image_feature_extractor" not in state_dict:
      _fix_old_state_dict(state_dict)

    super()._load_from_state_dict(
      state_dict, prefix, local_metadata, strict,
      missing_keys, unexpected_keys, error_msgs)

  def preload(self, examples, n_procs=None, pbar=True):
    self.image_feature_extractor.preload(examples, n_procs, pbar)


@dataclass
class GPV1CollateFn:
  is_train: bool
  tokenizer: PreTrainedTokenizer
  image_collate: ImageCollater
  word_to_idx: Dict
  max_text_len: int
  answering_type: str

  def encode_answers(self, targets):
    B = len(targets)
    answers = [''] * B
    for i, t in enumerate(targets):
      if 'answer' in t:
        answers[i] = t['answer']

    if self.answering_type == 'classification':
      padded_inputs = [None] * len(answers)
      padded_token_ids = [None] * len(answers)
      for i, answer in enumerate(answers):
        padded_inputs[i] = ['__cls__', answer]
        padded_token_ids[i] = []
        for token in padded_inputs[i]:
          if token in self.word_to_idx:
            token_id = self.word_to_idx[token]
          else:
            token_id = self.word_to_idx['__unk__']

          padded_token_ids[i].append(token_id)

      padded_token_ids = torch.LongTensor(padded_token_ids)

    elif self.answering_type == 'generation':
      padded_inputs = [None] * len(answers)
      S = 0
      for i, answer in enumerate(answers):
        if answer == '':
          sent = f'__cls__ __stop__'
        else:
          sent = f'__cls__ {answer} __stop__'
        padded_inputs[i] = [w.lower() for w in word_tokenize(sent)]
        S = max(S, len(padded_inputs[i]))

      padded_token_ids = [None] * len(answers)
      for i, padded_tokens in enumerate(padded_inputs):
        padded_tokens.extend(['__pad__'] * (S - len(padded_tokens)))
        token_ids = [None] * S
        for j in range(S):
          if padded_tokens[j] in self.word_to_idx:
            token_ids[j] = self.word_to_idx[padded_tokens[j]]
          else:
            token_ids[j] = self.word_to_idx['__unk__']

        padded_token_ids[i] = token_ids[:self.max_text_len]

      padded_token_ids = torch.LongTensor(padded_token_ids)

    else:
      raise NotImplementedError

    return padded_inputs, padded_token_ids

  def __call__(self, batch: List[GPVExample]):
    image_inputs, box_targets = self.image_collate.collate(batch)

    queries = []
    for ex in batch:
      if isinstance(ex.query, list):
        # Select a random query, this should only happen for captioning at
        # train time since those are given multiple queries
        q = ex.query[np.random.randint(0, len(ex.query))]
      else:
        q = ex.query
      queries.append(q)

    if self.tokenizer is not None:
      queries = self.tokenizer(
        queries,
        padding=True,
        return_tensors='pt'
      )

    out = dict(image_inputs=image_inputs, queries=queries)

    targets = [dict(
      answer=x.target_answer,
      task=full_task_name(x.task)
    ) if x.target_answer is not None else dict() for x in batch]
    answer_tokens, answer_token_ids = self.encode_answers(targets)

    for i, target in enumerate(targets):
      if box_targets[i] is not None:
        target.update(box_targets[i])
      target["answer"] = batch[i].target_answer
      target["task"] = full_task_name(batch[i].task)
      target['answer_token_ids'] = answer_token_ids[i, 1:]

    out["targets"] = targets
    out["answer_token_ids"] = answer_token_ids

    return out
