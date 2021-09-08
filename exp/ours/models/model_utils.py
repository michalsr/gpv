import logging
from typing import List, Dict, Tuple, Any, Set

from collections import Callable, defaultdict

from dataclasses import dataclass
from transformers import PreTrainedTokenizer, T5Tokenizer

from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.dataset import Task
from exp.ours.image_featurizer.image_featurizer import ImageCollater
from exp.ours.models.losses import GpvBatchLabels
from exp.ours.train.optimizer_builder import ParameterSet
import numpy as np

from exp.ours.util.nlp_utils import prepare_batch_from_pre_encoded


@ParameterSet.register("vision")
class VisionParameterExtractor(ParameterSet):
    def get(self, model):
      return list(p for p in model.image_feature_extractor.parameters() if p.requires_grad)


@ParameterSet.register("pretrained-vision-features")
class PretrainedVisionParameterExtractor(ParameterSet):
  def get(self, model):
    pretrained = list(model.image_feature_extractor.get_pretrained_parameters())
    total = len(pretrained)
    logging.info(f"Found {len(pretrained)} pretrained parameters of {total} in image feautre extractor")
    return list(p for p in pretrained if p.requires_grad)


@ParameterSet.register("backbone")
class BackboneParameterExtractor(ParameterSet):
  def get(self, model):
    fe = model.image_feature_extractor
    if hasattr(fe, "backbone"):
      backbone = fe.backbone
    else:
      backbone = fe.detr.backbone
    return list(p for p in backbone.parameters() if p.requires_grad)


@ParameterSet.register("detr")
class DetrParameterExtractor(ParameterSet):
  def get(self, model):
    detr = model.image_feature_extractor.detr
    in_backbone = set(id(x) for x in detr.backbone.parameters())
    return list(x for x in detr.parameters() if id(x) not in in_backbone)


@dataclass
class CollateWithTokenizer(Callable):
  tokenizer: PreTrainedTokenizer
  image_collater: ImageCollater
  q_len: int
  ans_len: int
  pre_tokenized: bool
  other_collate: Any = None

  def __call__(self, batch: List[GPVExample]):
    queries = []
    answers = []

    for ex in batch:
      q = ex.query[np.random.randint(0, len(ex.query))]

      if ex.target_answer is None or len(ex.target_answer) == 0:
        # This is a bit messy since it conflates no output text requested (therefore, a
        # detection examples) with an unlabelled example (predicting a caption with no known label),
        # although there is no harm done since we ignore the labels when predicting anyway
        if self.pre_tokenized:
          a = np.array([self.tokenizer.pad_token_id], dtype=np.int)
        else:
          a = self.tokenizer.pad_token
      elif isinstance(ex.target_answer, list):
        a = ex.target_answer[np.random.randint(0, len(ex.target_answer))]
      else:
        a = ex.target_answer

      if self.pre_tokenized:
        queries.append(q.tolist() + [self.tokenizer.eos_token_id])
        answers.append(a.tolist() + [self.tokenizer.eos_token_id])
      else:
        queries.append(q)
        answers.append(a)

    image_data = self.image_collater.collate(batch)
    image_inputs, box_targets = image_data

    if self.pre_tokenized:
      queries = prepare_batch_from_pre_encoded(
        queries, self.tokenizer, self.q_len, truncation=True)
      answers = prepare_batch_from_pre_encoded(
        answers, self.tokenizer, self.ans_len)
    else:
      queries = self.tokenizer(
        queries, return_tensors='pt', padding=True, max_length=self.q_len, truncation=True)
      answers = self.tokenizer(
        answers, return_tensors='pt', padding=True, max_length=self.ans_len)

    if any(x.segmentation_label is not None for x in batch):
      # raise NotImplementedError("Collate segmentation labels")
      segmentation_labels = [None for _ in batch]
    else:
      segmentation_labels = [None for _ in batch]

    labels = GpvBatchLabels(
      [x.task for x in batch],
      answers["input_ids"],
      box_targets,
      segmentation_labels=segmentation_labels
    )

    out = dict(
      input_ids=queries["input_ids"],
      input_mask=queries["attention_mask"],
      labels=labels,
      image_inputs=image_inputs
    )

    if self.other_collate:
      out.update(self.other_collate.collate(batch, out))
    return out
