import torch

import hydra
from allennlp.nn.beam_search import BeamSearch

from datasets.coco_multitask_dataset import CocoCaptioning as CocoCaptioningDataset
from datasets.coco_multitask_dataset import CocoVqa as CocoVqaDataset
from datasets.coco_multitask_dataset import CocoClassification as CocoClsDataset
from datasets.coco_multitask_dataset import CocoDetection as CocoDetectionDataset

from exp.gpv.models.gpv import GPV
from exp.ours.util import our_utils, py_utils
from exp.ours.data.dataset import GpvDataset, Task
from exp.ours.models.gpv1 import GPV1
from exp.ours.data.gpv_example import GPVExample
import numpy as np

from exp.ours.models.model import GPVOutput
from utils.detr_misc import collate_fn as detr_collate_fn


class OurGpvGetter:

  def __init__(self, cfg, seed, tasks, device, model):
    torch.manual_seed(seed)
    model = GPV1(cfg, _model=model)
    model.image_feature_extractor.initialize()
    model.eval()
    model.to(device)
    examples = {}
    for task in tasks:
      if task == Task.CAPTIONING:
        for ex in GpvDataset(task, "val").load():
          for e in model.preprocess_example_train(ex):
            examples[e.id] = e
      else:
        for ex in GpvDataset(task, "val").load():
          examples[ex.get_gpv_id()] = ex
    self.examples = examples
    self.model = model

  def get(self, example_ids, train_loss=True):
    examples = []
    for ex_id in example_ids:
      ex = self.examples[ex_id]
      if isinstance(ex, GPVExample):
        examples.append(ex)
      else:
        ex = self.model.preprocess_example_train(ex)
        assert len(ex) == 1
        examples.append(ex[0])
    batch = self.model.get_collate()(examples)
    assert not self.model.training
    if train_loss:
      return self.model(**batch)
    else:
      return self.model.predict(**batch)


class TheirGpvGetter:
  DATASETS = {
    Task.CAPTIONING: (CocoCaptioningDataset, "CocoCaptioning", "coco_captioning"),
    Task.VQA: (CocoVqaDataset, "CocoVqa", "coco_vqa"),
    Task.CLS: (CocoClsDataset, "CocoClassification", "coco_classification"),
    Task.DETECTION: (CocoDetectionDataset, "CocoDetection", "coco_detection")
  }

  def __init__(self, cfg, seed, tasks, device, model):
    torch.manual_seed(seed)
    model.eval()
    model.to(device)
    self.model = model
    self.id_to_ds = {}
    for task in tasks:
      ds_cls, ds_name, task_conf_name = self.DATASETS[task]
      ds = ds_cls(cfg.task_configs[task_conf_name], "val")
      for ix, ex in enumerate(ds.samples):
        if task == Task.DETECTION:
          gpv_id = f"coco-boxes{ex['image']['image_id']}-cat{ex['category_id']}"
        elif task == Task.CLS:
          gpv_id = f"coco-box{ex['id']}"
        elif task == Task.VQA:
          gpv_id = f"vqa{ex['question_id']}"
        elif task == Task.CAPTIONING:
          gpv_id = f"coco-cap{ex['cap_id']}"
        else:
          raise ValueError()
        self.id_to_ds[gpv_id] = (ds, ix)

  def get(self, example_ids, train_loss=True):
    model: GPV = self.model
    assert not self.model.training
    examples = []
    for ex in example_ids:
      ds, ix = self.id_to_ds[ex]
      examples.append(ds[ix])
    imgs, queries, targets = detr_collate_fn(examples)
    answer_tokens, answer_token_ids = model.encode_answers(targets)
    for i, t in enumerate(targets):
      t['answer_token_ids'] = answer_token_ids[i, 1:]
    if train_loss:
      loss, loss_dict = model(imgs, queries, answer_token_ids, targets, return_all_losses=True)
      return loss, loss_dict
    else:
      outputs, start_pred, initial_state, decode_fn = model.init_beam_search(imgs, queries)
      rel = outputs['pred_relevance_logits'].softmax(-1)[:, :, 0]
      boxes = outputs["pred_boxes"]
      return GPVOutput(
        boxes, rel,
        (start_pred, initial_state, decode_fn)
      )


@hydra.main(config_path=f'../../configs', config_name=f"exp/gpv")
def compare_evaluator_output(cfg):
  our_utils.add_stdout_logger()
  seed = 451614
  tasks = [
    Task.DETECTION,
  ]
  device = torch.device("cpu")

  print()
  print("Loading their GPV")
  other = TheirGpvGetter(cfg, seed, tasks, device)

  print()
  print("Loading our GPV")
  our = OurGpvGetter(cfg.model, seed, tasks, device)
  keys = list(our.examples)
  np.random.RandomState().shuffle(keys)
  examples = keys[:1]

  print()
  print("Loading state...")
  state_f = "/Users/chris/Programming/gpv/models/gpv-coco-sce/ckpts/model.pth"
  state = torch.load(state_f, map_location="cpu")["model"]
  state = {k[len("module."):]: v for k, v in state.items()}
  our.model.gpv.load_state_dict(state)
  other.model.load_state_dict(state)

  bs_out = BeamSearch(
    our.model.get_end_index(), max_steps=40, beam_size=5)
  print(examples)

  with torch.no_grad():
    print("Getting ours")
    out: GPVOutput = our.get(examples, False)
    # print(out.relevance)
    ix, probs = bs_out.search(*out.generation_initialization)
    print(ix[:, 0])

    print("Getting other")
    out: GPVOutput = other.get(examples, False)
    ix, probs = bs_out.search(*out.generation_initialization)
    print(ix[:, 0])


@hydra.main(config_path=f'../../configs', config_name=f"exp/gpv")
def compare_train_outputs(cfg):
  py_utils.add_stdout_logger()
  seed = 451614
  tasks = [
    # Task.CAPTIONING,
    # Task.VQA,
    Task.CLS,
    # Task.DETECTION
  ]
  device = torch.device("cpu")

  model = GPV(cfg.model)
  model.load_pretr_detr(device)

  print()
  print("Loading their GPV")
  other = TheirGpvGetter(cfg, seed, tasks, device, model)

  print()
  print("Loading our GPV")
  our = OurGpvGetter(cfg.model, seed, tasks, device, model)
  gpv = our.model.gpv


  keys = list(our.examples)
  np.random.RandomState(seed).shuffle(keys)
  examples = keys[:1]
  print(examples)

  with torch.no_grad():
    print("Getting ours")
    print(our.get(examples))

    print("Getting other")
    print(other.get(examples))


if __name__ == '__main__':
  compare_train_outputs()
