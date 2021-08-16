import hydra
import torch
from allennlp.nn.beam_search import BeamSearch

from torch.utils.data import DataLoader

from datasets.coco_datasets import CocoVqa, CocoCaptioning, CocoClassification, CocoDetection
from exp.gpv.metrics import vqa_accuracy, cap_metrics, cls_metrics, det_metrics
from exp.ours.util import our_utils, py_utils
from exp.ours.data.dataset import GpvDataset, Task
from exp.ours.models.gpv1 import GPV1
from exp.ours.models.model import GPVModel
from exp.ours.train.runner import run_model, CollateWithBatch, GPVExampleOutput
from exp.ours.data.source_data import load_instances, CocoCaptions
from exp.ours.train.evaluator import VqaEvaluator, CaptionEvaluator, ClsEvaluator, \
  LocalizationEvaluator

from utils.detr_misc import collate_fn as detr_collate_fn


NUM_WORKERS = 0


def other_eval(task: Task, model: GPVModel, cfg, sl):
  all_examples = load_instances(str(task), "val", True)
  examples = all_examples

  # for ex in examples:
  #   # 12057,
  #   if ex["image"]["image_id"] in {12902}:
  #     print(ex["answer"])
  # exit()

  if task == Task.VQA:
    examples.sort(key=lambda x: x["question_id"])
    examples = examples[sl]
    dataset = CocoVqa(cfg.task_configs.coco_vqa, "val", samples=examples)
  elif task == Task.CAPTIONING:
    examples.sort(key=lambda x: x["cap_id"])
    examples = examples[sl]
    dataset = CocoCaptioning(cfg.task_configs.coco_captioning, "val", samples=examples)
  elif task == Task.CLS:
    examples.sort(key=lambda x: x["id"])
    examples = examples[sl]
    dataset = CocoClassification(cfg.task_configs.coco_classification, "val", samples=examples)
  elif task == Task.DETECTION:
    examples.sort(key=lambda x: x["id"])
    examples = examples[sl]
    dataset = CocoDetection(cfg.task_configs.coco_detection, "val", samples=examples)
  else:
    raise NotImplementedError()

  device = our_utils.get_model_device(model)

  eval_dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=NUM_WORKERS,
    shuffle=False,
    collate_fn=detr_collate_fn)

  with torch.no_grad():
    if task == task.VQA:
      print(float(vqa_accuracy(model.gpv, eval_dataloader, cfg, device=device)))
    elif task == task.CLS:
      print(float(cls_metrics(model.gpv, eval_dataloader, cfg, device=device)))
    elif task == task.CAPTIONING:
      for k, v in cap_metrics(model.gpv, eval_dataloader, cfg, device=device, all_samples=all_examples).items():
        print(k, float(v))
    elif task == task.DETECTION:
      print(float(det_metrics(model.gpv, eval_dataloader, cfg, device=device)))
    else:
      raise NotImplementedError(task)


def our_eval(task: Task, model: GPVModel, sl):
  all_examples = GpvDataset(task, "val").load()
  examples = all_examples

  if task == Task.VQA:
    evaluator = VqaEvaluator()
    max_steps = 8
    examples.sort(key=lambda x: x.question_id)
  elif task == Task.CAPTIONING:
    out = []
    for ex in examples:
      for i, cap in enumerate(ex.captions):
        out.append(CocoCaptions((cap.id, ex.image_id), [cap], None))
    examples = out
    examples.sort(key=lambda x: x.image_id)
    examples = [CocoCaptions(ex.image_id[1], ex.captions) for ex in examples]

    # prep = []
    # for ex in examples:
    #   cap = ex.captions[0]
    #   prep.append(GPVExample(
    #     str(cap.id), Task.CAPTIONING, ex.image_id,
    #     cap.meta["gpv1-query"],
    #     target_boxes=None,
    #     target_box_labels=None,
    #     target_answer=cap.caption,
    #   ))

    evaluator = CaptionEvaluator()
    max_steps = 50
  elif task == Task.CLS:
    examples.sort(key=lambda x: x.id)
    evaluator = ClsEvaluator()
    max_steps = 5
  elif task == Task.DETECTION:
    examples.sort(key=lambda x: x.meta["gpv1-id"])
    evaluator = LocalizationEvaluator()
    max_steps = None
  else:
    raise NotImplementedError()

  examples = examples[sl]

  if max_steps is None:
    bs = None
  else:
    bs = BeamSearch(model.get_end_index(), max_steps=max_steps, beam_size=1)

  prepped = [model.preprocess_example(x) for x in examples]

  loader = DataLoader(
    prepped,
    batch_size=32,
    collate_fn=CollateWithBatch(model.get_collate()),
    num_workers=NUM_WORKERS,
    shuffle=False,
    sampler=None,
    pin_memory=False
  )
  pred = run_model(model, loader, bs, beams_to_keep=1)
  stats = evaluator.evaluate(all_examples, pred, allow_partial=True)
  for k, v in stats.items():
    print(str(k), float(v))


def dbg():

  original = {
    111: "a bunch of oranges in a glass bowl",
    112: "a giraffe walking in a fenced in area."
  }
  ours = {
    "coco-image-cap12902": 'a bunch of oranges in a glass bowl',
    "coco-image-cap12057": 'a giraffe walking in a fenced in area.'
  }


@hydra.main(config_path=f'../../configs', config_name=f"exp/gpv")
def main(cfg):
  py_utils.add_stdout_logger()

  print("Loading model")

  model = GPV1(cfg.model)

  print("Loading state...")
  state_f = cfg.eval.ckpt
  state = torch.load(state_f, map_location="cpu")["model"]
  state = {k[len("module."):]: v for k, v in state.items()}
  model.gpv.load_state_dict(state)

  model.eval()
  device = our_utils.get_device()
  model.to(device)

  sample = slice(5, 7)
  task = Task.CAPTIONING

  # print("Run other...")
  # other_eval(task, model, cfg, sample)
  #
  print("Run our...")
  our_eval(task, model, sample)


def test_tok():
  all_examples = GpvDataset(Task.CAPTIONING, "val").load()

  for i in range(3):
    out = CaptionEvaluator().evaluate(
      all_examples,
      {
        'coco-image-cap12902': GPVExampleOutput(None, None, ['a bunch of oranges in a glass bowl'], None),
        'coco-image-cap12057': GPVExampleOutput(None, None, ['a giraffe walking in a fenced in area.'], None)
       },
      allow_partial=True
    )
    print(out)


if __name__ == '__main__':
  main()