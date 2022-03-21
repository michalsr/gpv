import logging
from collections import defaultdict, Counter
from os.path import join, dirname

from data.coco.synonyms import SYNONYMS
from exp.ours import file_paths
from exp.ours.data.dataset import *
from exp.ours.models.model import PredictionArg
from exp.ours.util import image_utils, py_utils
from exp.ours.util.py_utils import int_to_str
from utils.io import load_json_object


def get_coco_categories():
  coco_file = join(dirname(__file__), "coco_categories.json")
  return load_json_object(coco_file)


COCO_ID_TO_CATEGORY = {x["id"]: x["name"] for x in get_coco_categories()}
COCO_CATEGORIES = list(COCO_ID_TO_CATEGORY.values())
COCO_CATEGORIES_TO_ID = {k: i for i, k in enumerate(COCO_CATEGORIES)}



def load_instances(kind, split,split_txt, gpv_split=True,unseen_split=False) -> List[Dict]:
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
  #split_txt = split_txt
  print(split_txt,'split txt')
  #split_txt = "gpv_split"
  # elif unseen_split==True:
  #   split_txt = "held_out_all"
  # else:
  #   split_txt = "original_split"
  print(ds,file_paths.SOURCE_DIR,split_txt)
  target_file = join(file_paths.SOURCE_DIR, ds, split_txt, f"{split}.json")
  logging.info(f"Loading instances from {target_file}")
  g = load_json_object(target_file)
  logging.info(f'{len(g)} total number of instances')
  return load_json_object(target_file)


def load_gpv_boxes(split, gpv_split,split_txt,raw_instances=None) -> List[LocalizationExample]:
  """Load GPV-I detection data"""
  if raw_instances == None:
    raw_instances = load_instances("detection", split, gpv_split,split_txt)
  out = []
  for x in raw_instances:
    if "coco_categories" in x:
      cats = x["coco_categories"]
      meta = {
        "gpv1-seen": cats["seen"],
        "gpv1-unseen": cats["unseen"],
        "gpv1-query": x["query"],
        "gpv1-id": x["id"]
      }
    else:
      meta = {
        "gpv1-query": x["query"],
        "gpv1-id": x["id"]
      }
    image_id = x["image"]["image_id"]
    cat_id = x["category_id"]
    gpv_id = f"coco-boxes{image_id}-cat{cat_id}"
    if 'category_name' in x:
      name = x['category_name']
    else:
      name =   COCO_ID_TO_CATEGORY[cat_id]
    if 'query' in x:
      q = x['query']
      meta['query'] = x['query']
    else:
      q= None
    bbox = LocalizationExample(
      gpv_id, x["image"]["image_id"], np.array(x["boxes"]),
      name, meta,query=q)
    out.append(bbox)
  return out


def load_gpv_vqa(split, gpv_split,split_txt,raw_instances=None) -> List[VqaExample]:
  """Load GPV-I VQA data"""
  if raw_instances == None:
    raw_instances = load_instances("vqa", split, gpv_split,split_txt)
  out = []
  for x in raw_instances:
    cats = x.get("coco_categories")
    if "answer" in x:
      meta = {"gpv1-answer": x["answer"]}
    else:
      meta = {}
    if cats is not None:
      meta.update({"gpv1-seen": cats["seen"], "gpv1-unseen": cats["unseen"], })
    if "all_answers" in x:
      answers = Counter(x["all_answers"])
    else:
      answers = None
    q = VqaExample(
      f"vqa{x['question_id']}", x["image"]["image_id"], x["query"],
      answers, meta=meta)
    out.append(q)
  return out


def load_gpv_captioning(split, gpv_split,split_txt,raw_instances=None) -> List[CaptioningExample]:
  """Load GPV-I captioning data"""
  if raw_instances == None:
    raw_instances = load_instances("cap", split, gpv_split,split_txt)
  grouped_by_image = defaultdict(list)
  for i, x in enumerate(raw_instances):
    meta = {}
    if "coco_categories" in x:
      cats = x["coco_categories"]
      meta.update({
        "gpv1-unseen": cats["unseen"],
        "gpv1-seen": cats["seen"],
      })
    if "answer" in x:
      meta["gpv1-answer"] = x["answer"]
    meta["gpv1-query"] = x["query"]
    if "cap_id" not in x:
      assert not gpv_split
      assert split == "test"
      cap_id = f"coco-cap-test{i}"
    else:
      cap_id = x["cap_id"]
    q = Caption(f"coco-cap{cap_id}", x.get("answer"), meta)
    grouped_by_image[x["image"]["image_id"]].append(q)

  out = []
  for image_id, captions in grouped_by_image.items():
    gpv_id = f"coco-image-cap{image_utils.get_coco_int_id(image_id)}"
    out.append(CaptioningExample(gpv_id, image_id, captions))
  return out


def load_gpv_cls(split, gpv_split,split_txt,raw_instances=None) -> List[ClsExample]:
  return _load_gpv_cls(split, gpv_split,split_txt,raw_instances)


def load_gpv_ident(split, gpv_split,split_txt,raw_instances=None) -> List[ClsExample]:
  return _load_gpv_cls(split, gpv_split, split_txt,raw_instances,True)


def _load_gpv_cls(split, gpv_split, split_txt,raw_instances=None,in_context=False) -> List:
  """Load GPV-I CLS data"""
  if in_context:
    def fn(i, image_id, category_id, box, meta):
      # TODO should change to `ident` to `cic` if we can avoid breaking already saved
      # prediction files
      return ClsExample(
        f"coco-ident{i}", Task.CLS_IN_CONTEXT, image_id,
        COCO_ID_TO_CATEGORY[category_id], query_box=box, meta=meta)
  else:
    def fn(i, image_id, category_id, box, meta):
      return ClsExample(
        f"coco-box{i}", Task.CLS, image_id,
        COCO_ID_TO_CATEGORY[category_id], crop=box, meta=meta)

  raw_instances = load_instances("cls", split, gpv_split,split_txt)
  out = []
  for x in raw_instances:
    cats = x.get("coco_categories")
    meta = {"gpv1-query": x["query"]}
    if cats is not None:
      meta.update({"gpv1-seen": cats["seen"], "gpv1-unseen": cats["unseen"]})
    assert x["answer"] == COCO_ID_TO_CATEGORY[x["category_id"]]
    q = fn(
      x["id"], x["image"]["image_id"], x["category_id"],
      x["boxes"], meta=meta)
    out.append(q)
  return out


def load_webqa(split,raw_instances=None) -> List[ClsExample]:
  """Load WebQA data"""
  if raw_instances == None:
    raw_instances = load_instances("webqa", split)
  out = []
  for x in raw_instances:
    meta = {"gpv1-query": x["query"], "bing-query": x["bing_query"],
            "gpv1-seen": x["coco_categories"]["seen"],
            "gpv1-unseen": x["coco_categories"]["unseen"],
            "question_type": x["question_type"],
            "rank": x["image"]["rank"]}
    q = ClsExample(f"webqa-{x['id']}", Task.WEBQA, x["image"]["image_id"], x["answer"], meta=meta)
    out.append(q)
  return out


GPV_KINDS = {
  'vqa': load_gpv_vqa,
  'cls': load_gpv_cls,
  'cap': load_gpv_captioning,
  'det': load_gpv_boxes,
  'webqa': load_webqa,
}


def load_gpv_instances(kind, split,split_txt,gpv_split=False,raw_instances=False):

  return GPV_KINDS[kind](split, split_txt, gpv_split=gpv_split,raw_instances=raw_instances)


def split_seen_unseen(instances):
  unseen_instances = []
  seen_instances = []
  for instance in instances:
    if isinstance(instance, CaptioningExample):
      unseen = sum(len(x.meta["gpv1-unseen"]) > 0 for x in instance.captions)
      unseen = unseen > 1
    else:
      unseen = instance.meta["gpv1-unseen"]
    if unseen:
      unseen_instances.append(instance)
    else:
      seen_instances.append(instance)
  return unseen_instances, seen_instances


@PredictionArg.register("coco-categories")
class CocoCategories(PredictionArg, list):
  def __init__(self, synonyms=False):
    self.synonyms = synonyms
    if self.synonyms:
      super().__init__(py_utils.flatten_list(SYNONYMS[x] for x in COCO_CATEGORIES))
    else:
      super().__init__(COCO_CATEGORIES)


@Dataset.register("gpv")
class GpvDataset(Dataset):
  KINDS = {
    Task.VQA: load_gpv_vqa,
    Task.CLS: load_gpv_cls,
    Task.CLS_IN_CONTEXT: load_gpv_ident,
    Task.CAPTIONING: load_gpv_captioning,
    Task.DETECTION: load_gpv_boxes,
  }

  UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass']
  UNSEEN2 = ['banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']

  UNSEEN_GROUPS = {
    Task.VQA: UNSEEN1,
    Task.CLS: UNSEEN2,
    Task.CLS_IN_CONTEXT: UNSEEN2,
    Task.CAPTIONING: UNSEEN1,
    Task.DETECTION: UNSEEN2,
  }

  def __init__(self, task: Task, split: str, gpv_split=True,
               sample=None, seen_sample=None, unseen_sample=None,
               per_example_captions=False,split_txt="gpv_split",raw_instances=None):
    if split not in {"test", "val", "train"}:
      raise ValueError(split)
    if sample is not None and (seen_sample is not None or unseen_sample is not None):
      raise ValueError("Cannot specify sample and seen/unseen sample")
    self.sample = sample
    self.task = task
    self.split = split
    self.gpv_split = gpv_split
    self.seen_sample = seen_sample
    self.unseen_sample = unseen_sample
    self.per_example_captions = per_example_captions
    self.split_txt = split_txt
    self.raw_instances = raw_instances

  def get_name(self):
    kind = "gpvsce" if self.gpv_split else "gpv"
    name = f"{kind}-{self.task}-{self.split}"
    if self.seen_sample is not None:
      name += f"-se{int_to_str(self.seen_sample)}"
    if self.unseen_sample is not None:
      name += f"-us{int_to_str(self.unseen_sample)}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def get_task(self) -> Task:
    return self.task

  def get_answer_options(self, synonyms=False):
    if self.task not in {Task.CLS, Task.CLS_IN_CONTEXT}:
      raise ValueError()
    return CocoCategories(synonyms)
  def change_split(self,new_split):
    self.split_txt = new_split
  def load(self):
    print(self.split,'split')
    print(self.gpv_split,'gpv split')
    print(self.split_txt,'split txt')
    instances = self.KINDS[self.task](self.split,self.split_txt,self.raw_instances)
    if self.per_example_captions and self.task == Task.CAPTIONING:
      per_ex = []
      for instance in instances:
        for cap in instance.captions:
          per_ex.append(CaptioningExample(cap.gpv_id, instance.image_id, [cap], cap.meta))
      instances = per_ex

    if self.seen_sample is not None or self.unseen_sample is not None:
      instances.sort(key=lambda x: x.gpv_id)
      np.random.RandomState(613423).shuffle(instances)
      unseen, seen = split_seen_unseen(instances)
      return unseen[:self.unseen_sample] + seen[:self.seen_sample]
    elif self.sample:
      instances.sort(key=lambda x: x.gpv_id)
      np.random.RandomState(613423).shuffle(instances)
      return instances[:self.sample]
    else:
      print(len(instances),'instances')
      return instances
