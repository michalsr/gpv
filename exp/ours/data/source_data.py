"""Functional interface to our "raw" data"""

import logging
import sys
from collections import defaultdict, Counter
from os.path import join, exists, dirname
from typing import List, Tuple, Optional, Dict, Any, Union

from dataclasses import dataclass

from exp.ours import file_paths
from exp.ours.util import image_utils
from utils.io import load_json_object, dump_json_object
import numpy as np


def get_coco_categories():
  coco_file = join(dirname(__file__), "coco_categories.json")
  return load_json_object(coco_file)


ID_TO_COCO_CATEGORY = {x["id"]: x["name"] for x in get_coco_categories()}
COCO_CATEGORY_TO_ID = {v: k for k, v in ID_TO_COCO_CATEGORY.items()}
COCO_CATEGORIES = [x[1] for x in sorted(ID_TO_COCO_CATEGORY.items(), key=lambda x: x[0])]


def load_instances(kind, split, gpv_split=True) -> List[Dict]:
  """Loads GPV-I in list-of-dictionary format"""

  if kind == "cls":
    ds = "coco_classification"
  elif kind == "vqa":
    ds = "vqa"
  elif kind in {"det", "detection", "coco_detection"}:
    ds = "coco_detection"
  elif kind in {"cap", "captioning", "coco_captions"}:
    ds = "coco_captions"
  elif kind in {"web_80"}:
    ds = "web_80"
  else:
    raise NotImplementedError(kind)
  if ds == "web_80":
    split_txt = ""
  elif gpv_split:
    split_txt = "gpv_split"
  else:
    split_txt = "original_split"
  target_file = join(file_paths.SOURCE_DIR, ds, split_txt, f"{split}.json")
  logging.info(f"Loading instances from {target_file}")
  return load_json_object(target_file)


@dataclass
class CocoBBoxes:
  """Image with all associated bounding boxes for a class"""

  @classmethod
  def from_json(cls, data):
    data = dict(data)
    data["bboxes"] = np.array(data["bboxes"])
    return cls(**data)

  image_id: str
  category_id: int
  bboxes: np.ndarray
  bbox_ids: List[int]
  categories: Optional[List[str]] = None
  meta: Dict[str, Any] = None
  _image_size: Optional[Tuple[int, int]] = None

  @property
  def category(self):
    return ID_TO_COCO_CATEGORY[self.category_id]

  @property
  def crop(self):
    return None

  @property
  def query_box(self):
    return None

  def to_json(self):
    return dict(
      image_id=self.image_id, category_id=self.category_id, bboxes=self.bboxes.tolist(),
      bbox_ids=self.bbox_ids
    )

  def get_gpv_id(self):
    return f"coco-boxes{image_utils.get_coco_int_id(self.image_id)}-cat{self.category_id}"


@dataclass(frozen=True)
class ClassWebClsExample:
  id: Union[id, str]
  image_id: str
  category_int: int
  category: str
  meta: Optional[Dict[str, Any]] = None

  @property
  def crop(self):
    return None

  @property
  def query_box(self):
    return None

  def get_gpv_id(self):
    return f"web-{self.image_id}"


@dataclass(frozen=True)
class WebQaExample:
  id: Union[id, str]
  image_id: str
  question: str
  category_int: int
  answer: str
  question_type: str
  meta: Optional[Dict[str, Any]] = None

  @property
  def crop(self):
    return None

  @property
  def query_box(self):
    return None

  def get_gpv_id(self):
    return f"web-{self.image_id}"


@dataclass(frozen=True)
class CocoBoxClsExample:
  """Example for classifying a single bbox in an image"""
  id: int
  image_id: str
  category_id: int
  box: Tuple[float, float, float, float]
  meta: Optional[Dict[str, Any]] = None

  @property
  def crop(self):
    return self.box

  @property
  def query_box(self):
    return None

  @property
  def category(self):
    return ID_TO_COCO_CATEGORY[self.category_id]

  def get_gpv_id(self):
    return f"coco-box{self.id}"


@dataclass(frozen=True)
class CocoBoxIdentificationExample:
  """Example for identifying a single bbox in an image"""
  id: int
  image_id: str
  category_id: int
  box: Tuple[float, float, float, float]
  meta: Optional[Dict[str, Any]] = None

  @property
  def crop(self):
    return None

  @property
  def query_box(self):
    return self.box

  @property
  def category(self):
    return ID_TO_COCO_CATEGORY[self.category_id]

  def get_gpv_id(self):
    return f"coco-ident{self.id}"


@dataclass(frozen=True)
class CocoCaption:
  """Image caption"""

  id: int
  caption: str
  meta: Optional[Dict[str, Any]] = None

  def get_gpv_id(self):
    return f"coco-cap{self.id}"

  @property
  def crop(self):
    return None

  @property
  def query_box(self):
    return None


@dataclass(frozen=True)
class CocoCaptions:
  """Image with all associated captions"""

  image_id: str
  captions: List[CocoCaption]
  meta: Optional[Dict[str, Any]] = None

  def get_gpv_id(self):
    return f"coco-image-cap{image_utils.get_coco_int_id(self.image_id)}"

  @property
  def crop(self):
    return None

  @property
  def query_box(self):
    return None


@dataclass(frozen=True)
class VqaQuestion:
  """Question and image VQA pair"""

  @staticmethod
  def from_json(data):
    return VqaQuestion(
      data["question_id"],
      data["image_id"],
      data["question"],
      sys.intern(data["question_type"]),
      sys.intern(data["answer_type"]),
      answers=Counter({sys.intern(k): v for k, v in data["answers"].items()})
    )

  question_id: int
  image_id: str
  question: str
  question_type: str
  answer_type: str
  answers: Counter
  meta: Optional[Dict[str, Any]] = None

  def to_json(self):
    return dict(
      question_id=self.question_id,
      question=self.question,
      image_id=self.image_id,
      question_type=self.question_type,
      answer_type=self.answer_type,
      answers={k: v for k, v in self.answers.items()}
    )

  def get_gpv_id(self):
    return f"vqa{self.question_id}"

  @property
  def crop(self):
    return None

  @property
  def query_box(self):
    return None


def load_gpv_boxes(split, gpv_split) -> List[CocoBBoxes]:
  """Load GPV-I detection data"""

  raw_instances = load_instances("detection", split, gpv_split)
  out = []
  for x in raw_instances:
    cats = x["coco_categories"]
    meta = {
      "gpv1-seen": cats["seen"],
      "gpv1-query": x["query"],
      "gpv1-unseen": cats["unseen"],
      "gpv1-id": x["id"]
    }
    bbox = CocoBBoxes(
      x["image"]["image_id"], x["category_id"], np.array(x["boxes"]), x["instance_ids"],
      cats["seen"] + cats["unseen"], meta=meta)
    out.append(bbox)
  return out


def load_gpv_vqa(split, gpv_split) -> List[VqaQuestion]:
  """Load GPV-I VQA data"""

  raw_instances = load_instances("vqa", split, gpv_split)
  out = []
  for x in raw_instances:
    cats = x.get("coco_categories")
    meta = {"gpv1-answer": x["answer"]}
    if cats is not None:
      meta.update({"gpv1-seen": cats["seen"], "gpv1-unseen": cats["unseen"], })
    q = VqaQuestion(
      x["question_id"], x["image"]["image_id"], x["query"],
      x["anno"]["question_type"], x["anno"]["answer_type"],
      Counter(x["all_answers"]),
      meta=meta)
    out.append(q)
  return out


def load_gpv_captioning(split, gpv_split) -> List[CocoCaptions]:
  """Load GPV-I captioning data"""

  raw_instances = load_instances("cap", split, gpv_split)
  grouped_by_image = defaultdict(list)
  for x in raw_instances:
    cats = x["coco_categories"]
    meta = {
      "gpv1-unseen": cats["unseen"],
      "gpv1-seen": cats["seen"],
      "gpv1-answer": x["answer"],
      "gpv1-query": x["query"]
    }
    q = CocoCaption(x["cap_id"], x["answer"], meta)
    grouped_by_image[x["image"]["image_id"]].append(q)

  out = []
  for k, v in grouped_by_image.items():
    out.append(CocoCaptions(k, v))
  return out


def load_gpv_cls(split, gpv_split) -> List[CocoBoxClsExample]:
  return _load_gpv_cls(split, gpv_split, False)


def load_gpv_ident(split, gpv_split) -> List[CocoBoxIdentificationExample]:
  return _load_gpv_cls(split, gpv_split, True)


def _load_gpv_cls(split, gpv_split, identification=False) -> List:
  """Load GPV-I CLS data"""
  if identification:
    def fn(i, image_id, category_id, box, meta):
      return CocoBoxIdentificationExample(i, image_id, category_id, box, meta)
  else:
    fn = CocoBoxClsExample

  raw_instances = load_instances("cls", split, gpv_split)
  out = []
  for x in raw_instances:
    cats = x.get("coco_categories")
    meta = {"gpv1-query": x["query"]}
    if cats is not None:
      meta.update({"gpv1-seen": cats["seen"], "gpv1-unseen": cats["unseen"]})
    assert x["answer"] == ID_TO_COCO_CATEGORY[x["category_id"]]
    q = fn(
      x["id"], x["image"]["image_id"], x["category_id"],
      x["boxes"], meta=meta)
    out.append(q)
  return out


def load_webqa(split) -> List[WebQaExample]:
  """Load WebQA data"""
  fn = WebQaExample
  raw_instances = load_instances("webqa", split)
  out = []
  for x in raw_instances:
    meta = {"gpv1-query": x["query"], "bing-query": x["bing_query"],
            "gpv1-seen": x["coco_categories"]["seen"],
            "gpv1-unseen": x["coco_categories"]["unseen"],
            "rank": x["image"]["rank"]}
    q = fn(
      x["id"], x["image"]["image_id"], x["category_id"],
      x["answer"], x["question_type"], meta=meta)
    out.append(q)
  return out


GPV_KINDS = {
  'vqa': load_gpv_vqa,
  'cls': load_gpv_cls,
  'cap': load_gpv_captioning,
  'det': load_gpv_boxes,
  'webqa': load_webqa,
}


def load_gpv_instances(kind, split, gpv_split):
  return GPV_KINDS[kind](split, gpv_split)


def load_coco_boxes(file) -> List[CocoBBoxes]:
  """Load original coco detection data"""

  data = load_json_object(file)

  grouped_by_image = defaultdict(list)
  for anno in data['annotations']:
    key = (anno["image_id"], anno["category_id"])
    grouped_by_image[key].append((anno["id"], anno["bbox"]))

  out = []
  for (image_id, cat_id), boxes in grouped_by_image.items():
    ids = [x[0] for x in boxes]
    boxes = np.array([x[1] for x in boxes])
    out.append(CocoBBoxes(image_id, cat_id, boxes, ids))
  return out


def load_coco_captions(file) -> List[CocoCaptions]:
  """Load original coco captioning data"""

  data = load_json_object(file)
  grouped_by_image = defaultdict(list)
  for anno in data['annotations']:
    grouped_by_image[anno["image_id"]].append(
      CocoCaption(CocoCaption(anno["id"], anno["caption"])))

  out = []
  for image_id, captions in grouped_by_image.items():
    out.append(CocoCaptions(image_id, captions))
  return out


def load_vqa_questions(questions, annotations) -> List[VqaQuestion]:
  """Load original VQA data"""

  annotation_json = load_json_object(annotations)
  annotations = {x["question_id"]: x for x in annotation_json['annotations']}
  assert len(annotation_json['annotations']) == len(annotations)

  questions = load_json_object(questions)["questions"]
  assert len(questions) == len(annotations)

  out = []
  for question in questions:
    anno = annotations[question["question_id"]]
    answers = Counter()
    answers.update(x["answer"] for x in anno["answers"])
    out.append(VqaQuestion(
      anno["question_id"], anno["image_id"], question["question"],
      anno["question_type"], anno["answer_type"], answers))
  return out


def get_coco_boxes(split: str, cache=True):
  """Load original coco detection data"""

  if split not in {"train", "val"}:
    raise ValueError(split)

  cache_name = join(file_paths.CACHE_DIR, f"coco-bboxes-{split}-v3.json")
  if exists(cache_name) and cache:
    logging.info(f"Loading {split} coco bboxes from cache")
    data = load_json_object(cache_name, compress=False)
    return [CocoBBoxes.from_json(x) for x in data]

  logging.info(f"Loading {split} coco bbox files")
  raw_file = join(file_paths.COCO_ANNOTATIONS, f"instances_{split}2014.json")
  boxes = load_coco_boxes(raw_file)

  logging.info(f"Saving to cache")
  if cache:
    dump_json_object([x.to_json() for x in boxes], cache_name, compress=False)

  return boxes


def get_coco_captions(split: str):
  """Load original coco captioning data"""

  if split not in {"train", "val"}:
    raise ValueError(split)
  logging.info(f"Loading {split} coco captions")
  raw_file = join(file_paths.COCO_ANNOTATIONS, f"captions_{split}2014.json")
  return load_coco_captions(raw_file)


def get_vqa2_questions(split: str):
  if split not in {"train", "val"}:
    raise ValueError(split)

  cache_name = join(file_paths.CACHE_DIR, f"vqa2.0-{split}.json")
  if exists(cache_name):
    logging.info(f"Loading {split} vqa2.0 from cache")
    data = load_json_object(cache_name, compress=False)
    return [VqaQuestion.from_json(x) for x in data]

  logging.info(f"Loading {split} vqa2.0 questions")
  anno_file = join(file_paths.VQA2_SOURCE, f"v2_mscoco_{split}2014_annotations.json")
  q_file = join(file_paths.VQA2_SOURCE, f"v2_OpenEnded_mscoco_{split}2014_questions.json")
  questions = load_vqa_questions(q_file, anno_file)

  logging.info(f"Saving to cache")
  dump_json_object([x.to_json() for x in questions], cache_name, compress=False)

  return questions

