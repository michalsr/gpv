from collections import defaultdict
from numbers import Number

import torch
from allennlp.common import FromParams, Registrable
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

import third_party.detection_metrics.lib.Evaluator as det_evaluator
from data.coco.synonyms import SYNONYMS
from exp.ours.data.gpv_data import GPVExample

from exp.ours.data.source_data import CocoCaptions, VqaQuestion, CocoBoxClsExample, CocoBBoxes
from exp.ours.util.image_utils import get_image_size
from exp.ours.train.runner import GPVExampleOutput

from typing import Dict, Optional, List, Counter

from dataclasses import dataclass

from exp.ours.util import py_utils
import numpy as np

from exp.ours.train.quiet_ptbtokenizer import QuitePTBTokenizer


def vqa_score(answer, ground_truth_answer_counts):
  gt_answers = {k.lower(): v for k, v in ground_truth_answer_counts.items()}
  return min(gt_answers.get(answer, 0) / 3, 1)


@dataclass(frozen=True)
class ResultKey(FromParams):
  metric_name: str
  subset_name: Optional[str] = None
  dataset_name: Optional[str] = None

  def __str__(self):
    out = [self.dataset_name, self.subset_name, self.metric_name]
    return "/".join(x for x in out if x is not None)

  def __repr__(self):
    return str(self)


class Evaluator(Registrable):
  """Computes evaluations metrics"""

  def evaluate(
      self, examples: List, predictions: Dict[str, GPVExampleOutput],
      allow_partial=False, subset_mapping=None
  ) -> Dict[ResultKey, Number]:
    """Computes corpus wide metrics

    :param examples: List of source examples
    :param predictions: example key -> model output
    :param allow_partial: Allow the predictions to only cover a subset of `examples`,
                          in which only those predictions should be evaluated
    :param subset_mapping: Function that maps example -> list of strings, names of the subsets that
                           example is part of
    """
    raise NotImplementedError()


class PerExampleEvaluator(Evaluator):
  """Computes per-examples evaluations metrics"""

  def evaluate_examples(self, examples: List, predictions: Dict[str, GPVExampleOutput])-> List[Dict[str, Number]]:
    raise NotImplementedError()

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, GPVExampleOutput],
      allow_partial=False,
      mean=True,
      subset_mapping=None
  ) -> Dict[ResultKey, Number]:
    examples_with_predictions = [x for x in examples if x.get_gpv_id() in predictions]
    if not allow_partial and (len(examples) != len(examples_with_predictions)):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions
    per_example_scores = self.evaluate_examples(examples, predictions)
    per_metric_scores = py_utils.transpose_list_of_dicts(per_example_scores)

    subsets = defaultdict(list)
    all_ids = [x.get_gpv_id() for x in examples]

    id_to_ix = {k: i for i, k in enumerate(all_ids)}
    subsets[None] = list(range(len(all_ids)))

    if subset_mapping is not None:
      for example in examples:
        example_id = id_to_ix[example.get_gpv_id()]
        for subset in subset_mapping(example):
          subsets[subset].append(example_id)

    out = {}

    for metric_name, score in per_metric_scores.items():
      score = np.array(score)
      for subset_name, ixs in subsets.items():
        if mean:
          out[ResultKey(metric_name, subset_name)] = float(np.mean(score[ixs]))
        else:
          out[ResultKey(metric_name, subset_name)] = (float(np.sum(score[ixs])), len(ixs))

    return out


@Evaluator.register("vqa-evaluator")
class VqaEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[VqaQuestion],
                        predictions: Dict[str, GPVExampleOutput], add_scores=False):
    out = []
    for example in examples:
      answer = predictions[example.get_gpv_id()].text[0]
      score = vqa_score(answer, example.answers)
      out.append(dict(score=score))
    return out


@Evaluator.register("cls-evaluator")
class ClsEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[CocoBoxClsExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      answer = predictions[example.get_gpv_id()].text[0].lower()
      gt_answer = SYNONYMS[example.category]
      out.append(dict(accuracy=answer in gt_answer))
    return out


@Evaluator.register("webqa-evaluator")
class WebQaEvaluator(PerExampleEvaluator):
  def evaluate_examples(self, examples: List[GPVExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      answer = predictions[example.get_gpv_id()].text[0].lower()
      gt_answer = SYNONYMS[example.target_answer] if example.target_answer in SYNONYMS else [example.target_answer]
      out.append(dict(accuracy=answer in gt_answer))
    return out


@Evaluator.register("detect-evaluator")
class DetectionEvaluator(PerExampleEvaluator):

  def __init__(self, iou_thresh=0.5):
    self.iou_thresh = iou_thresh

  def evaluate_examples(self, examples: List[CocoBBoxes], predictions: Dict[str, GPVExampleOutput]):
    eval_engine = det_evaluator.Evaluator()
    out = []

    for i, ex in enumerate(examples):
      pred = predictions[ex.get_gpv_id()]
      scores = pred.relevance
      pred_boxes = pred.boxes.copy()
      gt_boxes = np.array(ex.bboxes)

      # Convert cx cy, w, h -> x1, y1, w, h
      pred_boxes[:, 0] = pred_boxes[:, 0] - 0.5 * pred_boxes[:, 2]
      pred_boxes[:, 1] = pred_boxes[:, 1] - 0.5 * pred_boxes[:, 3]

      # convert to relative coordinates
      W, H = get_image_size(ex.image_id)
      gt_boxes[:, 0] = gt_boxes[:, 0] / W
      gt_boxes[:, 1] = gt_boxes[:, 1] / H
      gt_boxes[:, 2] = gt_boxes[:, 2] / W
      gt_boxes[:, 3] = gt_boxes[:, 3] / H

      B = pred_boxes.shape[0]
      all_boxes = det_evaluator.BoundingBoxes()
      for b in range(B):
        x, y, w, h = pred_boxes[b]
        all_boxes.addBoundingBox(det_evaluator.BoundingBox(
          imageName=ex.image_id,
          classId=ex.category,
          x=x,
          y=y,
          w=w,
          h=h,
          typeCoordinates=det_evaluator.CoordinatesType.Relative,
          imgSize=(W, H),
          bbType=det_evaluator.BBType.Detected,
          classConfidence=scores[b],
          format=det_evaluator.BBFormat.XYWH))

      B = gt_boxes.shape[0]
      for b in range(B):
        x, y, w, h = gt_boxes[b]
        all_boxes.addBoundingBox(det_evaluator.BoundingBox(
          imageName=ex.image_id,
          classId=ex.category,
          x=x,
          y=y,
          w=w,
          h=h,
          typeCoordinates=det_evaluator.CoordinatesType.Relative,
          imgSize=(W, H),
          bbType=det_evaluator.BBType.GroundTruth,
          format=det_evaluator.BBFormat.XYWH))

      det_metrics = eval_engine.GetPascalVOCMetrics(all_boxes, self.iou_thresh)
      out.append({"AP": det_metrics[0]['AP']})

    return out


class CachingPTBTokenizer:

  def __init__(self, do_cache=True):
    self.tokenizer = QuitePTBTokenizer()
    self._tokenize_cache = {}
    self.do_cache = do_cache

  def tokenize_sentences(self, sentences: List[str]):
    out = [None for _ in range(len(sentences))]
    to_tokenize = {}
    for ix, sent in enumerate(sentences):
      tok_cached = self._tokenize_cache.get(sent)
      if tok_cached is not None and self.do_cache:
        out[ix] = tok_cached
      else:
        to_tokenize[str(ix)] = [dict(caption=sent)]

    tok_out = self.tokenizer.tokenize(to_tokenize)

    for k, caps in tok_out.items():
      source_sent = to_tokenize[k][0]["caption"]
      if self.do_cache:
        self._tokenize_cache[source_sent] = caps[0]
      out[int(k)] = caps[0]
    assert not any(x is None for x in out)
    return out

  def tokenize_captions(self, *sets):
    keys = []
    sents = []
    for i, caption_set in enumerate(sets):
      for key, captions in caption_set.items():
        for caption in captions:
          assert isinstance(caption, str)
          sents.append(caption)
          keys.append((i, key))

    tokenized = self.tokenize_sentences(sents)
    out = [defaultdict(list) for _ in sets]

    for (caption_set, key), tokens in zip(keys, tokenized):
      out[caption_set][key].append(tokens)

    return [dict(x) for x in out]


def get_per_caption_data(examples, predictions):
  caption_examples = []
  caption_predictions = {}
  for ex in examples:
    pred = predictions[ex.get_gpv_id()]
    for cap in ex.captions:
      caption_examples.append(CocoCaptions(cap.id, ex.captions))
      caption_predictions[caption_examples[-1].get_gpv_id()] = pred
  return caption_examples, caption_predictions


@Evaluator.register("cap-evaluator")
class CaptionEvaluator(Evaluator):
  def __init__(self, cider=True, bleu=4,
               caching_tokenizer=True, per_caption=False):
    self.cider = cider
    self.bleu = bleu
    self.per_caption = per_caption
    self.caching_tokenizer = caching_tokenizer
    scorers = {}
    if cider:
      # from exp.ours.eval.fast_cider import FastCider
      scorers["cider"] = Cider()
    if bleu:
      scorers["bleu"] = Bleu(bleu)
    self.scorers = scorers
    self.tokenizer = CachingPTBTokenizer(caching_tokenizer)

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, GPVExampleOutput],
      allow_partial=False,
      subset_mapping=None,
  ):
    examples_with_predictions = [x for x in examples if x.get_gpv_id() in predictions]
    if not allow_partial and (len(examples) != len(examples_with_predictions)):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions

    if self.per_caption:
      examples, predictions = get_per_caption_data(examples, predictions)

    subsets = defaultdict(list)
    subsets[None] = examples
    if subset_mapping is not None:
      for example in examples:
        example_subsets = subset_mapping(example)
        for subset in example_subsets:
          subsets[subset].append(example)

    out = {}
    for subset_name, examples in subsets.items():
      all_scores = self._get_scores(examples, predictions)

      results = {}
      for name, scorer in self.scorers.items():
        corpus_scores, _ = all_scores[name]
        if isinstance(scorer, Cider):
          results["cider"] = corpus_scores
        elif isinstance(scorer, Bleu):
          scores, _ = all_scores[name]
          for i, score in enumerate(corpus_scores):
            results[f"bleu{i+1}"] = score
      if subset_name is not None:
        results["n"] = len(examples)
      out.update({ResultKey(metric_name=k, subset_name=subset_name): v for k, v in results.items()})
    return out

  def evaluate_examples(self, examples: List[CocoCaptions], predictions: Dict[str, GPVExampleOutput]):
    all_scores = self._get_scores(examples, predictions)

    per_examples_scores = [{} for _ in examples]
    for name, scorer in self.scorers.items():
      score, scores = all_scores[name]
      if isinstance(scorer, Cider):
        for score, ex_scores in zip(scores, per_examples_scores):
          ex_scores["cider"] = score
      elif isinstance(scorer, Bleu):
        scores = py_utils.transpose_lists(scores)
        for score, ex_scores in zip(scores, per_examples_scores):
          for i, s in enumerate(score):
            ex_scores[f"bleu{i+1}"] = s

    return per_examples_scores

  def _get_scores(self,  examples: List[CocoCaptions], predictions: Dict[str, GPVExampleOutput]):
    res = {}
    gts = {}
    for ix, instance in enumerate(examples):
      key = instance.get_gpv_id()
      assert key not in res
      res[key] = [predictions[instance.get_gpv_id()].text[0]]
      gts[key] = [x.caption.lower() for x in instance.captions]

    res, gts = self.tokenizer.tokenize_captions(res, gts)

    scores = {}

    for name, scorer in self.scorers.items():
      if isinstance(scorer, Bleu):
        scores[name] = scorer.compute_score(gts, res, verbose=0)
      else:
        scores[name] = scorer.compute_score(gts, res)
    return scores
