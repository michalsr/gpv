import logging
from typing import List, Dict, Tuple, Any, Set

from collections import Callable, defaultdict

from dataclasses import dataclass

from torch.jit import Error
from transformers import PreTrainedTokenizer, T5Tokenizer
import pdb 
from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.dataset import Task
from exp.ours.image_featurizer.image_featurizer import ImageCollater
from exp.ours.models.losses import GpvBatchLabels
from exp.ours.train.optimizer_builder import ParameterSet
import numpy as np
from exp.ours.util import our_utils, py_utils, image_utils
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

  def __call__(self, batch):
    query_save = []
    answers = []
    indicies = []
    mil_answers = []
    queries = []
    syn_ids = []
    new_batch = []
    image_ids = []
    for b in batch:
 
      if type(b) == list:
        for element in b:
      
          new_batch.append(element)
      else:
        new_batch.append(b)
    batch = new_batch

    #print(batch,'batch')
    #print('hello')
    #print(type(batch[0]),'batch type')
    # if type(batch[0]) == list:
    #   #pdb.set_trace()
    #   #print(batch[0],'batc 0')

    #   #print(batch[0],'batch')

    #   new_batch = py_utils.flatten_list(batch)

    #   # if batch[0][0].task == (Task.IMAGECONTRAST or Task.TEXTCONTRAST or Task.SYNONYM):
    #   #   new_batch = [item for sublist in batch nfor item in sublist]
   
    #   # if batch[0][0].task == Task.SYNONYM:
      
    #   #   for i in range(len(batch)):
    #   #     new_batch.append(batch[i][0])
    #   #     new_batch.append(batch[i][1])
    #   #     print(new_batch[-1].image_id,'1')
    #   #     print(new_batch[-2].image_id,'2')
    #   batch = new_batch

     
    #   # if batch[0].task == Task.IMAGECONTRAST or batch[0].task == Task.TEXTCONTRAST:
    #   #   idx = batch[0].index_of_class
    #   #   idxes = [int(x.index_of_class) for x in batch]
    #   #   for y in batch:
    #   #     if int(y.index_of_class) != int(idx):
    #   #       print(idxes,'idxes')
    #   #       raise ValueError
    #   #   print('Images check out')
    #   # if batch[0].task == (Task.SYNONYM):
    #   #   #image id should be the same for every pair 
    #   #   for i in range(0,len(batch),2):
    #   #     if str(batch[i].image_id) != str(batch[i+1].image_id):
    #   #       print(batch[i].image_id,batch[i+1].image_id)
    #   #       raise ValueError 
    #     # print(int(idx)==int(batch[1].index_of_class))
    #     # if not all(idxes) == int(idx):
    #     #   print(idxes)
    #     #   raise Error
     
    #   # for i,b in enumerate(new_batch):
    #   #   print(i,b.index_of_class,'index of class')
    # # for i,ex in enumerate(batch):
    # #   if i!= len(batch)-2:
    # #     print(batch[i].image_id,batch[i+1].image_id)
    # #     if str(batch[i].image_id) != str(batch[i+1].image_id):
    # #       print('BAD')
    # #       break
    # # for i in range(len(batch)):
    # #   if i<len(batch)-3:
    # #     print(batch[i].image_id,'1 again')
    # #     print(batch[i+1].image_id,'2 again')
    # #print(batch[0].task,'batch task')
    # #print(len(batch),'batch size')
    
    for ex in batch:

      # print(ex.image_id,'image id')
      # print(ex.index_of_class,'index of class')
      #print(ex.target_answer,'target answer')
      query_save.append(ex.relevance_query)
      image_ids.append(ex.image_id)
      if ex.correct_answer!= None:
        mil_answers.append(ex.correct_answer)
      #print('Appended indicies')
      if ex.task == Task.SYNONYM:
        syn_ids.append(ex.syn_id)
      else:
        syn_ids.append(None)
 

      if ex.task == Task.IMAGECONTRAST:

        indicies.append(ex.index_of_class)
      else:
        indicies.append(None)
      q = ex.query[np.random.randint(0, len(ex.query))]
      #print(ex.target_answer == None,'target answer')
      #print(ex.target_answer,'taget answer')
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
    #print(len(queries),'query length 1')

    image_data = self.image_collater.collate(batch)
    #print('Collated image data')
    image_inputs, box_targets = image_data

    #print(len(box_targets),'box targets')
    #print(image_inputs.size(),'image inputs size')
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
    #print(len(batch),'batch size here'[])
    labels = GpvBatchLabels(
      [x.task for x in batch],
      answers["input_ids"],
      box_targets,
      segmentation_labels=segmentation_labels,index_of_class=indicies,mil_labels=mil_answers,syn_id=syn_ids,image_ids=image_ids,queries=query_save
    )
    #print('Gathered labels')
    #print(labels.index_of_class,'index of class collate')
    #print(labels,'labels')
    out = dict(
      input_ids=queries["input_ids"],
      input_mask=queries["attention_mask"],
      labels=labels,
      image_inputs=image_inputs
    )

    if self.other_collate:
      out.update(self.other_collate.collate(batch, out))
    #print(out,'out')
    return out
