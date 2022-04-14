import re
from collections import defaultdict
from numbers import Number

import spacy
import torch
from allennlp.common import FromParams, Registrable
from nltk import word_tokenize
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval 
import third_party.detection_metrics.lib.Evaluator as det_evaluator
from data.coco.synonyms import SYNONYMS
from exp.ours import file_paths
from exp.ours.data.coco_segmentation import SegmentationExample
from exp.ours.data.gpv import COCO_CATEGORIES
from exp.ours.data.gpv_example import GPVExample

from exp.ours.data.dataset import VqaExample, CaptioningExample, ClsExample, LocalizationExample
from exp.ours.data.opensce import OPENSCE_SYNONYMS
from exp.ours.data.webqa import WebQaExample
from exp.ours.train.vqa2_eval_data import *
from exp.ours.util.image_utils import get_image_size
from exp.ours.train.runner import GPVExampleOutput

from typing import Dict, Optional, List, Counter

from dataclasses import dataclass, replace

from exp.ours.util import py_utils
import numpy as np

from exp.ours.train.quiet_ptbtokenizer import QuitePTBTokenizer
from utils.io import load_json_object, dump_json_object
from nltk.stem import WordNetLemmatizer
from collections import Counter

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
    print(len(predictions),'predictions')
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


@Evaluator.register("seg-evaluator")
class SegmentationEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[SegmentationExample],
                        predictions: Dict[str, GPVExampleOutput], add_scores=False):
    raise NotImplementedError()


@Evaluator.register("vqa-evaluator")
class VqaEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[VqaExample],
                        predictions: Dict[str, GPVExampleOutput], add_scores=False):
    out = []
    for example in examples:
      answer = predictions[example.gpv_id].text[0]
      print('new entry')
      print(answer,'answer')
      print(example.answers,'example answer')
      score = vqa_score(answer, example.answers)
      out.append(dict(score=score))
    return out


@Evaluator.register("cls-evaluator")
class ClsEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[ClsExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      answer = predictions[example.gpv_id].text[0].lower()
      #print(answer,'answer')
      gt_answer = SYNONYMS[example.category]
      #print(gt_answer,'gt answer')
      out.append(dict(accuracy=answer in gt_answer))
      #print(out,'out')
    return out


@Evaluator.register("webqa-evaluator")
class WebQaEvaluator(PerExampleEvaluator):
  def evaluate_examples(self, examples: List[WebQaExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      answer = predictions[example.get_gpv_id()].text[0].lower()
      gt_answer = SYNONYMS[example.answer] if example.answer in SYNONYMS else [example.answer]
      out.append(dict(accuracy=answer in gt_answer))
    return out


@Evaluator.register("opensce-cls")
class OpenSceClsEvaluator(PerExampleEvaluator):

  def __init__(self, top_k: Optional[List[int]]=(5,), use_synonyms=True):
    self.top_k = top_k
    self.use_synonyms = use_synonyms

  def evaluate_examples(self, examples: List[ClsExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      answers = [x.lower() for x in predictions[example.get_gpv_id()].text]
      # TODO following OpenSCE, we should use this when making predictions
      # instead of duing evaluations
      if self.use_synonyms:
        gt = OPENSCE_SYNONYMS.get(example.category, [example.category])
      else:
        gt = [example.category]
      vals = {"accuracy": answers[0] in gt}
      if self.top_k is not None:
        for k in self.top_k:
          assert len(answers) >= k
          vals[f"top{k}-acc"] = any(a in gt for a in answers[:k])
      out.append(vals)
    return out



@Evaluator.register("boosting-opensce-cls")
class BoostingOpenSceClsEvaluator(PerExampleEvaluator):
  def __init__(self, search_range, coco_syn, top_n, sub_eval: OpenSceClsEvaluator):
    self.search_range = search_range
    self.sub_eval = sub_eval
    self.coco_syn = coco_syn
    self.top_n = top_n
    self.coco_answer = set(COCO_CATEGORIES)
    if self.coco_syn:
      self.coco_answer = set(py_utils.flatten_list(SYNONYMS[x] for x in self.coco_answer))

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, GPVExampleOutput],
      allow_partial=False,
      mean=True,
      subset_mapping=None
  ) -> Dict[ResultKey, Number]:
    if mean is False:
      raise ValueError()

    if self.top_n is not None:
      for k, v in predictions.items():
        if len(v.text) < self.top_n:
          raise ValueError(f"Example has {len(v.text)} beams, but top n is {self.top_n}")
      predictions = {k: v.set_beams_to_keep(self.top_n) for k, v in predictions.items()}
    all_scores = []
    for ex in examples:
      p = predictions[ex.get_gpv_id()]
      coco_ixs = []
      for i, a in enumerate(p.text):
        if a in self.coco_answer:
          coco_ixs.append(i)
      coco_ixs = np.array(coco_ixs)
      scores = []
      if len(coco_ixs) > 0:
        for th in self.search_range:
          boosted = np.array(p.text_logprobs)
          boosted[coco_ixs] -= th
          answer = p.text[np.argmax(boosted)]
          scores.append(answer == ex.category)
      else:
        scores = [p.text[0] == ex.category] * len(self.search_range)
      all_scores.append(scores)
    all_scores = np.array(all_scores).mean(0)
    best_th_ix = np.argmax(all_scores)
    best_th = self.search_range[best_th_ix]

    revised = {}

    for k, p in predictions.items():
      boosted = np.array(p.text_logprobs)
      for i, a in enumerate(p.text):
        if a in self.coco_answer:
          boosted[i] -= best_th
      re_sort = np.argsort(-boosted)
      boosted = boosted[re_sort]
      revised[k] = replace(p, text_logprobs=boosted, text=[p.text[i] for i in re_sort])

    stats = {ResultKey(metric_name="boost"): best_th}
    for k, v in self.sub_eval.evaluate(examples, revised, subset_mapping=subset_mapping).items():
      stats[replace(k, metric_name=f"boost-{k.metric_name}")] = v
    stats.update(self.sub_eval.evaluate(examples, predictions, subset_mapping=subset_mapping))
    return {k: v for k, v in stats.items()}


def compute_vqa_accuracy(
    gt_answers: List[str],
    pred_answers: List[str]) -> List[float]:
  ngt_answers = [preprocess_answer(ans) for ans in gt_answers]
  topk_npred_answers = [preprocess_answer(ans) for ans in pred_answers]
  gt_consensus = Counter(ngt_answers)
  return [vqa_accuracy(ans, gt_consensus) for ans in topk_npred_answers]


def vqa_accuracy(npred_answer: str, gt_consensus: Counter):
  return min(gt_consensus[npred_answer]/3,1)


def processPunctuation(inText):
  outText = inText
  for p in punct:
    if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
      outText = outText.replace(p, '')
    else:
      outText = outText.replace(p, ' ')
  outText = periodStrip.sub("",outText,re.UNICODE)
  return outText


def processDigitArticle(inText):
  outText = []
  tempText = inText.lower().split()
  for word in tempText:
    word = manualMap.setdefault(word, word)
    if word not in articles:
      outText.append(word)
    else:
      pass
  for wordId, word in enumerate(outText):
    if word in contractions:
      outText[wordId] = contractions[word]
  outText = ' '.join(outText)
  return outText


def preprocess_answer(ans):
  ans = ans.replace('\n', ' ')
  ans = ans.replace('\t',' ')
  ans = ans.lower().strip()
  return processDigitArticle(processPunctuation(ans))


@Evaluator.register("opensce-vqa")
@Evaluator.register("opensce-vqa-v0")
class OpenSceVqaEvaluatorV0(PerExampleEvaluator):

  nlp = spacy.load('en_core_web_sm')
  lemmatizer = WordNetLemmatizer()

  @staticmethod
  def answer_match_iou(cand_tokens: List[str], ref_tokens: List[str]) -> float:
    cset = set(cand_tokens)
    rset = set(ref_tokens)
    intersection = len(cset.intersection(rset))
    union = len(cset.union(rset))
    return intersection / union

  @staticmethod
  def answer_match(cand_tokens: List[str], ref_tokens: List[str]) -> bool:
    return ' '.join(cand_tokens) == ' '.join(ref_tokens)

  @staticmethod
  def compute_accuracy(
      gt_answer: str,
      pred_answers: List[str],
      k: int) -> float:
    gt_tokens = OpenSceVqaEvaluator.get_tokens(gt_answer)
    return float(any(
      [OpenSceVqaEvaluator.answer_match(gt_tokens, OpenSceVqaEvaluator.get_tokens(ans)) for ans in pred_answers[:k]]))

  def __init__(self, top_k: Optional[List[int]]=(5,), spacy_lemmatizer=True):
    self.top_k = top_k
    self.spacy_lemmatizer = spacy_lemmatizer

  def get_tokens(self, txt: str):
    if self.spacy_lemmatizer:
      lemmas = [t.lemma_ for t in OpenSceVqaEvaluator.nlp(txt)]
    else:
      lemmas = [OpenSceVqaEvaluator.lemmatizer.lemmatize(x) for x in word_tokenize(txt)]
    articles = ['a', 'an', 'the']
    return [l for l in lemmas if l not in articles]

  def evaluate_example(self, example: VqaExample, pred: str):
    answer = self.get_tokens(pred.lower())
    gt = self.get_tokens(example.answers.lower())
    return OpenSceVqaEvaluator.answer_match_iou(gt, answer)

  def evaluate_examples(self, examples: List[VqaExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      max_k = max(self.top_k) if self.top_k else 1
      answers = [self.get_tokens(x.lower()) for x in predictions[example.get_gpv_id()].text[:max_k]]
      gt = self.get_tokens(example.answers.lower())
      vals = dict(
        acc=gt == answers[0],
        iou=OpenSceVqaEvaluator.answer_match_iou(gt, answers[0])
      )
      if self.top_k:
        for k in self.top_k:
          vals[f"top{k}-acc"] = float(any(x == gt for x in answers[:k]))
          vals[f"top{k}-iou"] = max(OpenSceVqaEvaluator.answer_match_iou(x, gt) for x in answers[:k])
      out.append(vals)
    return out


@Evaluator.register("opensce-vqa-v1")
class OpenSceVqaEvaluator(PerExampleEvaluator):

  def __init__(self, top_k: Optional[List[int]]=(5,)):
    self.top_k = top_k
    if top_k is not None:
      assert all(x > 0 for x in top_k)

  def evaluate_examples(self, examples: List[VqaExample], predictions: Dict[str, GPVExampleOutput]):
    out = []
    for example in examples:
      max_k = 1 if self.top_k is None else max(self.top_k)
      answers = predictions[example.get_gpv_id()].text[:max_k]
      gt = example.answers
      scores = compute_vqa_accuracy(gt, answers)
      vals = dict(acc=scores[0])
      if self.top_k:
        for k in self.top_k:
          vals[f"top{k}-acc"] = max(scores[:k])
      out.append(vals)
    return out


@Evaluator.register("localization-evaluator")
@Evaluator.register("detect-evaluator")
class LocalizationEvaluator(PerExampleEvaluator):

  def __init__(self, iou_thresh=0.5):
    self.iou_thresh = iou_thresh
  def evaluate_examples(self, examples: List[LocalizationExample], predictions: Dict[str, GPVExampleOutput],
                        return_pr=False):
    eval_engine = det_evaluator.Evaluator()
    out = []

    for i, ex in enumerate(examples):
      pred = predictions[ex.gpv_id]
      scores = pred.relevance
      pred_boxes = pred.boxes.copy()
      gt_boxes = np.array(ex.bboxes)

      # Convert cx cy, w, h -> x1, y1, w, h
      pred_boxes[:, 0] = pred_boxes[:, 0] - 0.5 * pred_boxes[:, 2]
      pred_boxes[:, 1] = pred_boxes[:, 1] - 0.5 * pred_boxes[:, 3]

      B = pred_boxes.shape[0]
      all_boxes = det_evaluator.BoundingBoxes()
      W, H = get_image_size(ex.image_id)

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

      normalized_gt = all(all(val <= 1.0 for val in b) for b in gt_boxes)
      if not normalized_gt:
        # convert to relative coordinates
        # TODO its a bit of hack to check this by looking coordinates > 1.0
        # but we need this check atm since DCE stores relative scaling
        # coco uses absolute
        W, H = get_image_size(ex.image_id)
        gt_boxes[:, 0] = gt_boxes[:, 0] / W
        gt_boxes[:, 1] = gt_boxes[:, 1] / H
        gt_boxes[:, 2] = gt_boxes[:, 2] / W
        gt_boxes[:, 3] = gt_boxes[:, 3] / H

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
      if return_pr:
        out.append(det_metrics[0])
      else:
        out.append({"AP": det_metrics[0]['AP']})

    return out
  def log_detection_eval_metrics(self,json_dataset, coco_eval):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    print(ind_lo,'ind lo')
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    print(coco_eval.eval['precision'].shape,'shape')
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(
        '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
            IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    category_ids = json_dataset.getCatIds()
    cats = [c['name'] for c in json_dataset.loadCats(category_ids)]

    for cls_ind, cls in enumerate(cats):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        print(cls_ind,'cls ind')
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi ), :, cls_ind, 0, 2]
        ap = np.mean(precision[precision > -1])
        print('{:.1f}'.format(100 * ap))
    print('~~~~ Summary metrics ~~~~')


  
  def evaluate_examples_coco(self,result_file_name,predictions=None):
    coco_dataset = COCO(annotation_file='/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase_coco/test.json')
    result_api = coco_dataset.loadRes(f'/data/michal5/gpv/learning_phase_data/coco_detection/{result_file_name}.json')   
    coco_eval = COCOeval(coco_dataset,result_api,'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    self.log_detection_eval_metrics(coco_dataset, coco_eval)
    return []





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


def get_per_caption_data(examples: List[CaptioningExample], predictions):
  # In per-caption evaluation the model makes one prediction for each ground truth
  # caption, each of which it still evaluated against all the captions,
  caption_examples = []
  caption_predictions = {}
  for ex in examples:
    pred = predictions[ex.gpv_id]
    for cap in ex.captions:
      caption_examples.append(CaptioningExample(cap.gpv_id, ex.image_id, ex.captions))
      caption_predictions[cap.gpv_id] = pred
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

  def evaluate_examples(self, examples: List[CaptioningExample], predictions: Dict[str, GPVExampleOutput]):
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

  def _get_scores(self,  examples: List[CaptioningExample], predictions: Dict[str, GPVExampleOutput]):
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

