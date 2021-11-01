from allennlp.common import Params
from dataclasses import replace

from exp.ours.data.coco_segmentation import SegmentationExample
from exp.ours.data.gpv_example import GPVExample, SegmentationLabel
from exp.ours.data.dataset import *
import torchvision.transforms as T
import numpy as np
from exp.ours.data.image_contrast import ImageContrastExample
from exp.ours.data.webqa import WebQaExample
from exp.ours.data.webqa_templates import WebQaQueryGenerator

NORMALIZE_TRANSFORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_stocastic_transforms(task: Task, cls_horizontal_flip=True):
  if task in {task.CLS, Task.CLS_IN_CONTEXT, Task.WEBQA,Task.IMAGECONTRAST}:
    transforms = [
      T.RandomApply([
        T.ColorJitter(0.4, 0.4, 0.4, 0.1)
      ], p=0.8)
    ]
    if cls_horizontal_flip:
      transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.RandomGrayscale(p=0.2))
  elif task == task.DETECTION:
    transforms = [
      T.RandomApply([
        T.ColorJitter(0.4, 0.4, 0.4, 0.1)
      ], p=0.8),
      T.RandomGrayscale(p=0.2),
    ]
  elif task == task.VQA or task == task.CAPTIONING:
    transforms = [
      T.RandomApply([
        T.ColorJitter(0.2, 0.2, 0.2, 0.0)
      ], p=0.8),
    ]
  else:
    raise NotImplementedError(task)
  return transforms


def get_train_transforms(task: Task, cls_horizontal_flip=True):
  if task in {Task.SEGMENTATION}:
    return None
  return T.Compose(
    [
      T.ToPILImage(mode='RGB')
    ] + get_stocastic_transforms(task, cls_horizontal_flip) +
    [
      T.ToTensor(),
      NORMALIZE_TRANSFORM
    ]
  )


def get_eval_transform():
  return T.Compose([
    T.ToPILImage(mode='RGB'),
    T.ToTensor(),
    NORMALIZE_TRANSFORM
  ])


CAPTION_QUERIES = [
  'Generate a caption.',
  'Generate a description.',
  'Describe this image.',
  'Describe the image.',
  'Caption this image.',
  'Caption the image.',
  'What is happening in this image?',
  'What is happening in the image?',
  'What is going on in this image?',
  'What is going on in the image?',
  'Generate a caption for this image.',
  'Generate a caption for the image.',
  'Generate a description for this image.',
  'Generate a description for the image.',
]

BBOX_QUERIES = [
  'Locate {}.',
  'Locate {} in the image.',
  'Locate {} in this image.',
  'Locate instances of {}.',
  'Locate instances of {} in the image.',
  'Locate instances of {} in this image.',
  'Locate all instances of {}.',
  'Locate all instances of {} in the image.',
  'Locate all instances of {} in this image.',
  'Find {}.',
  'Find {} in the image.',
  'Find {} in this image.',
  'Find instances of {}.',
  'Find instances of {} in the image.',
  'Find instances of {} in this image.',
  'Find all instances of {}.',
  'Find all instances of {} in the image.',
  'Find all instances of {} in this image.',
]

SEGMENTATION_QUERIES = [
  'Segment {}.',
  'Segment {} in the image.',
  'Segment {} in this image.',
  'Segment instances of {}.',
  'Segment instances of {} in the image.',
  'Segment instances of {} in this image.',
  'Segment all instances of {}.',
  'Segment all instances of {} in the image.',
  'Segment all instances of {} in this image.',
  'Find the masks of {}.',
  'Find the masks of {} in the image.',
  'Find the masks of {} in this image.',
]

CLS_QUERIES = [
  'What is this?',
  'What is this object?',
  'What object is this?',
  'What is this thing?'
]


class Gpv1Preprocessor(FromParams):

  def __init__(self, webqa_templates: Optional[WebQaQueryGenerator]=None, webqa_cls=False,
               relevance_query=None):
    self.webqa_templates = webqa_templates
    self.preprocess_text = None
    self.cls_queries_tok = None
    self.caption_queries_tok = None
    self.relevance_query = relevance_query
    self.webqa_cls = webqa_cls
    self.expand_cls = webqa_cls
    self._cache = {}

  def init(self, preprocess_text):
    self.preprocess_text = preprocess_text
    cls = CLS_QUERIES
    if self.webqa_cls:
      raise NotImplementedError()
    self.cls_queries_tok = [preprocess_text(x) for x in cls]
    self.caption_queries_tok = [preprocess_text(x) for x in CAPTION_QUERIES]

  def preprocess_example(self, example, is_train=False, include_query_box=False, include_meta=False):
    if include_query_box:
      default_query_box = np.array([[0.0, 0.0, 1.0, 1.0]])
    else:
      default_query_box = None

    # TODO ideally normalize boxes here too
 
    if isinstance(example,ImageContrastExample):
      answer = self.preprocess_text(example.answer)
      final_answer = np.append(answer,int(example.answer))
      #print(self.preprocess_text(example.answer).append('1'),'example answer')
      out = [GPVExample(
        example.gpv_id, example.task, example.image_id,
        query=[self.preprocess_text(example.query)],
        target_answer=final_answer,meta=example.contrast_group,index_of_class=example.answer
      )]
    elif isinstance(example, CaptioningExample):
      if is_train:
        out = []
        for cap in example.captions:
          out.append(GPVExample(
            cap.gpv_id,
            Task.CAPTIONING,
            example.image_id,
            self.caption_queries_tok,
            None,
            query_boxes=default_query_box,
            target_answer=[self.preprocess_text(cap.caption)],
            meta=cap.meta if include_meta else None,index_of_class=None
          ))
      else:
        out = [GPVExample(
          example.gpv_id,
          Task.CAPTIONING,
          example.image_id,
          self.caption_queries_tok,
          None,
          query_boxes=default_query_box,
          target_answer=[self.preprocess_text(x.caption) for x in example.captions if x.caption is not None],
          meta=example.meta if include_meta else None,index_of_class=None
        )]
    elif isinstance(example, SegmentationExample):
      out = [GPVExample(
        example.gpv_id,
        Task.SEGMENTATION,
        example.image_id,
        [self.preprocess_text(x.format(example.category)) for x in SEGMENTATION_QUERIES],
        segmentation_label=SegmentationLabel(
          example.iscrowd, example.area, example.segmentation
        ),index_of_class=None
      )]
    elif isinstance(example, LocalizationExample):
      if self.relevance_query is None:
        rel_query = None
      elif self.relevance_query == "category":
        rel_query = example.category
      elif self.relevance_query == "relevant":
        rel_query = "relevant"
      else:
        raise NotImplementedError(self.relevance_query)

      out = [GPVExample(
        example.gpv_id,
        Task.DETECTION,
        example.image_id,
        [self.preprocess_text(x.format(example.category)) for x in BBOX_QUERIES],
        example.bboxes,
        relevance_query=rel_query,
        query_boxes=default_query_box,
        target_answer=None,
        meta=example.meta if include_meta else None,index_of_class=None
      )]
    elif isinstance(example, VqaExample):
      if isinstance(example.answers, Counter):
        answer = max(example.answers.items(), key=lambda x: (x[1], len(x[0])))[0]
      else:
        answer = example.answers
      out = [GPVExample(
        example.gpv_id,
        Task.VQA,
        example.image_id,
        [self.preprocess_text(example.question)],
        query_boxes=default_query_box,
        target_answer=None if answer is None else self.preprocess_text(answer),
        meta=example.meta if include_meta else None,index_of_class=None
      )]
    elif isinstance(example, WebQaExample):
      query = [self.preprocess_text(x) for x in self.webqa_templates.get_prompts(example, is_train)]
      out = [GPVExample(
        example.gpv_id, example.task, example.image_id,
        query,
        target_answer=None if example.answer is None else self.preprocess_text(example.answer),index_of_class=None
      )]
    elif isinstance(example, GPVExample):
      # Currently assume the query and answer are just text
      assert isinstance(example.query, str)
      assert example.target_answer is None or isinstance(example.target_answer, str)
      out = [replace(
        example,
        query=[self.preprocess_text(example.query)],
        target_answer=None if example.target_answer is None else self.preprocess_text(example.target_answer),
        meta=None if include_meta else example.meta,
        query_boxes=example.query_boxes if example.query_boxes is not None else default_query_box,index_of_class=None
      )]
    elif isinstance(example, ClsExample):
      out = [GPVExample(
        example.gpv_id,
        example.task,
        example.image_id,
        self.cls_queries_tok,
        None,
        query_boxes=default_query_box if example.query_box is None else np.array([example.query_box]),
        crop=example.crop,
        target_answer=self.preprocess_text(example.category),
        meta=example.meta if include_meta else None,index_of_class=None
      )]
    else:
      raise NotImplementedError(example)

    if is_train:
      return out
    else:
      assert len(out) == 1
      return out[0]
