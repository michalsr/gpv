from allennlp.common import FromParams

from exp.ours.data.gpv_data import GPVExample, Task
from exp.ours.data.source_data import CocoCaptions, CocoBBoxes, CocoBoxClsExample, VqaQuestion, \
  CocoBoxIdentificationExample
import torchvision.transforms as T
import numpy as np

NORMALIZE_TRANSFORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_stocastic_transforms(task: Task, cls_horizontal_flip=True):
  if task == task.CLS or task == Task.CLS_IN_CONTEXT:
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
  'What is happening in this image.',
  'What is happening in the image.',
  'What is going on in this image.',
  'What is going on in the image.',
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

CLS_QUERIES = [
  'What is this?',
  'What is this object?',
  'What object is this?',
  'What is this thing?'
]


class Gpv1Preprocessor(FromParams):

  def __init__(self):
    self.preprocess_text = None
    self.cls_queries_tok = None
    self.caption_queries_tok = None

  def init(self, preprocess_text):
    self.preprocess_text = preprocess_text
    self.cls_queries_tok = [preprocess_text(x) for x in CLS_QUERIES]
    self.caption_queries_tok = [preprocess_text(x) for x in CAPTION_QUERIES]

  def preprocess_example(self, example, is_train=False, include_query_box=False):
    if include_query_box:
      all_image_box = np.array([[0.0, 0.0, 1.0, 1.0]])
    else:
      all_image_box = None

    if isinstance(example, CocoCaptions):
      if is_train:
        out = []
        for cap in example.captions:
          out.append(GPVExample(
            cap.get_gpv_id(),
            Task.CAPTIONING,
            example.image_id,
            self.caption_queries_tok,
            None,
            query_boxes=all_image_box,
            target_answer=[self.preprocess_text(cap.caption)]
          ))
      else:
        out = [GPVExample(
          example.get_gpv_id(),
          Task.CAPTIONING,
          example.image_id,
          self.caption_queries_tok,
          None,
          query_boxes=all_image_box,
          target_answer=[self.preprocess_text(x.caption) for x in example.captions]
        )]
    elif isinstance(example, CocoBBoxes):
      out = [GPVExample(
        example.get_gpv_id(),
        Task.DETECTION,
        example.image_id,
        [self.preprocess_text(x.format(example.category)) for x in BBOX_QUERIES],
        example.bboxes,
        query_boxes=all_image_box,
        target_answer=None
      )]
    elif isinstance(example, VqaQuestion):
      answer = max(example.answers.items(), key=lambda x: (x[1], len(x[0])))[0]
      out = [GPVExample(
        example.get_gpv_id(),
        Task.VQA,
        example.image_id,
        [self.preprocess_text(example.question)],
        query_boxes=all_image_box,
        target_answer=self.preprocess_text(answer)
      )]
    elif isinstance(example, (CocoBoxClsExample, CocoBoxIdentificationExample)):
      out = [GPVExample(
        example.get_gpv_id(),
        Task.CLS if isinstance(example, CocoBoxClsExample) else Task.CLS_IN_CONTEXT,
        example.image_id,
        self.cls_queries_tok,
        None,
        query_boxes=all_image_box if example.query_box is None else np.array([example.query_box]),
        crop=example.crop,
        target_answer=self.preprocess_text(example.category)
      )]
    else:
      raise NotImplementedError(example)

    if is_train:
      return out
    else:
      assert len(out) == 1
      return out[0]
