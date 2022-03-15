from typing import List, Union, Any, Optional, Tuple, Dict, Callable

import numpy as np
import torch
import torchvision.ops
from allennlp.common import Registrable
from allennlp.nn.beam_search import StepFunctionType, StateType
from dataclasses import dataclass
from torch import nn
import utils.io as io
from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.dataset import Task
import os 

@dataclass
class GPVExampleOutput:
  """GPV output for an example"""

  boxes: Union[torch.Tensor, np.ndarray, None]
  """[n_boxes, 4] box output in cxcywh format, normalized between 0 and 1"""

  relevance: Union[torch.Tensor, np.ndarray, None]
  """[n_boxes] Relevance score of each box, between 0 and 1"""

  text: Optional[List[str]]
  """top ranked text answers, sorted by score"""

  text_logprobs: Optional[List[float]]
  """score of each text answer"""
  image_ids: Optional[List]
  queries: Optional[List]


  def set_beams_to_keep(self, n):
    if self.text is None:
      return self
    return GPVExampleOutput(self.boxes, self.relevance, self.text[:n], self.text_logprobs[:n])


class PredictionArg(Registrable):
  """Generic super-type for typed prediction arguements"""
  pass


class GPVModel(nn.Module, Registrable):

  def initialize(self, load_params=True):
    """Initialize the model, used before training but not if loading a state dict

    This give the model a chance to load pre-trained parameters or pre-compute dataset
    statistics that do not need to be loaded if loading from a state_dict

    if `load_params` is false, the model should still set up all its parameters and buffers,
    but does not need to fill with the initialized values.
    """
    raise NotImplementedError()

  def get_collate(self, is_train=False) -> Callable[[List[GPVExample]], Dict[str, Any]]:
    """Function that maps pre-processed examples to tensors suitable for `forward`

    The returned function might need to be distributed across multiple worker processes,
    so implementors should return a light weight object rather than the a method
    bound to `self`
    """
    raise NotImplementedError()

  def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes the loss any scalars to log using the outputs of `self.get_collate()(batch)`
    """
    raise NotImplementedError()

  def predict(self, *args, **kwargs) -> List[GPVExampleOutput]:
    """Computes the test-time example outputs for a batch of examples"""
    raise NotImplementedError()

  def set_prediction_args(
      self, *args: Union[str, int, float, PredictionArg],
      **kwargs: Union[str, int, float, PredictionArg]
  ):
    """Sets parameters used during prediction"""
    raise NotImplementedError()

  def preprocess_example_train(self, example) -> List[GPVExample]:
    """Convert a training example for a task into a universal/pre-processed format

    We support a one-to-many mapping for train examples
    """

    # By default, use the general preprocessing method
    return [self.preprocess_example(example)]

  def preprocess_example(self, example) -> GPVExample:
    """Convert an eval example for a task into a universal/pre-processed format"""
    raise NotImplementedError()

def convert_to_total_output(gpv_examples):
    total_output = {'pred_boxes':[],'rel':[],'images':[],'queries':[]}
    for ex in gpv_examples:
      total_output['pred_boxes'].append(ex.boxes.tolist())
      total_output['rel'].append(ex.relevance.tolist())
      total_output['images'].append(ex.image_ids)
      total_output['queries'].append(ex.queries)
    print(total_output,'total output')
    return total_output

def build_per_example_output(text, text_scores, boxes, rel,image_ids,queries,json_output=None,n_boxes=None, box_format="cxcywh") -> List[GPVExampleOutput]:
  out = []
  #print(text,'text')
  print(image_ids,'image ids')
  if text_scores is not None:
    if isinstance(text_scores, torch.Tensor):
      text_scores = text_scores.cpu().numpy()

  if boxes is None:
    for txt, sc in zip(text, text_scores):
      out.append(GPVExampleOutput(None, None, txt, sc))
    return out

  if boxes.size()[:2] != rel.size():
    raise ValueError("Boxes and relevance have incompatible shapes")

  boxes = boxes.cpu()
  rel = rel.cpu().numpy()
  n_boxes = None if n_boxes is None else n_boxes.cpu().numpy()

  n = len(boxes)
  for i in range(n):
    if text is None:
      example_text, example_text_scores = None, None
    else:
      example_text, example_text_scores = text[i], text_scores[i]

    end = None if n_boxes is None else n_boxes[i]
    ixs = np.argsort(rel[i, :end])
    example_boxes = torchvision.ops.box_convert(boxes[i, ixs], box_format, "cxcywh")
 
     
    

    out.append(GPVExampleOutput(
      example_boxes.numpy(), rel[i, ixs], example_text, example_text_scores,image_ids[i],queries[i]
    ))

  return out

