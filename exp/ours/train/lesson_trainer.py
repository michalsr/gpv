import json
import logging
import math
import os
import re
import socket
from collections import defaultdict, Counter
from datetime import datetime
from os import mkdir, makedirs, getpid
from time import perf_counter
import random
import gc
import pdb
from sklearn.metrics import log_loss
from torch.jit import Error 
import h5py
import torch
from numbers import Number
from os.path import join, exists, dirname
from typing import List, Optional, Dict, Any, Union, Tuple
from exp.ours.data.image_contrast import ImageContrastDataset
from allennlp.common import FromParams, Params, Registrable
from allennlp.common.util import lazy_groups_of
from dataclasses import dataclass, replace
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from exp.ours.data.webqa import WebQaDataset
from torch import distributed as dist
from exp.ours.train.runner import BeamSearchSpec, DataLoaderBuilder
from exp.ours.train import evaluator
from exp.ours import file_paths
from exp.ours.data.stratified_subset_sampler import StratifiedSubsetSampler,ImageContrastSampler, SynonymSampler
from exp.ours.util import our_utils, py_utils, image_utils
from exp.ours.data.dataset import Dataset, Task
from exp.ours.models.model import GPVModel
from exp.ours.util.our_utils import SubsetSampler, DistributedSubsetSampler, select_run_dir
from exp.ours.util.py_utils import clear_if_nonempty, duration_to_str
from exp.ours.train.runner import BeamSearchSpec, run_model, CollateWithBatch, \
  GPVExampleOutput, PredictionArg, save_gpv_output, DataLoaderBuilder
from exp.ours.util.to_params import to_params
from exp.ours.train.evaluator import ResultKey, CaptionEvaluator, Evaluator
from exp.ours.train.optimizer_builder import OptimizerBuilder, TrainingScheduleBuilder
from utils.io import dump_json_object, load_json_object
import numpy as np
from exp.ours.data.dataset import GPV1_TASKS, GPV2_TASKS, Task
from exp.ours.data.gpv import GpvDataset, CocoCategories
from exp.ours.train.new_lr_scheduler import ReduceLROnPlateau

def get_datasets():
  tasks = {}  # Use a dictionary to preserve ordering
  tasks.update({x: None for x in GPV2_TASKS})
  train_datasets = []
  eval_datasets = []
  for task in tasks:
    train_datasets.append(TrainerDataset(GpvDataset(task, "train", True,unseen_split=True), str(task) + "-tr"))
    eval_datasets.append(TrainerDataset(GpvDataset(task, "val", True,unseen_split=True), str(task) + "-val"))

  best_model_key = [
    evaluator.ResultKey("accuracy", dataset_name="cls-val"),
    evaluator.ResultKey("accuracy", dataset_name="cic-val"),
    evaluator.ResultKey("score", dataset_name="vqa-val"),
    evaluator.ResultKey("cider", dataset_name="cap-val"),
    evaluator.ResultKey("AP", dataset_name="det-val"),
    evaluator.ResultKey("accuracy", dataset_name="webqa-val"),
  ]
  qtypes = WebQaDataset.QTYPES_NAME_TO_TYPES.get("basic", ("basic",))
  best_model_key = [x for x in best_model_key if any(x.dataset_name.startswith(str(t)) for t in tasks)]
  for x in train_datasets:
      if x.dataset.get_task() == Task.CAPTIONING:
        x.eval_sample = 5000
      else:
        x.eval_sample = 8000
  for x in eval_datasets:
      if x.dataset.get_task() == Task.CAPTIONING:
        x.eval_sample = 8000
      else:
        x.eval_sample = 12000
  webqa_train = WebQaDataset("train",
                                None, qtypes)
  evaluation = {
    Task.VQA: EvaluationSetup(
      evaluator.VqaEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 10))
    ),
    Task.CAPTIONING: EvaluationSetup(
      evaluator.CaptionEvaluator(per_caption=True),
      dict(beam_search_spec=BeamSearchSpec(1, 30))
    ),
    Task.DETECTION: EvaluationSetup(
      evaluator.LocalizationEvaluator(),
      dict(beam_search_spec=None)
    ),
    Task.CLS: EvaluationSetup(
      evaluator.ClsEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 5), answer_options=CocoCategories())
    ),
    Task.CLS_IN_CONTEXT: EvaluationSetup(
      evaluator.ClsEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 5), answer_options=CocoCategories())
    ),
    Task.WEBQA: EvaluationSetup(
      evaluator.WebQaEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 5),
           answer_options=webqa_train.get_answer_options(False))
    )
  }
  other_log = {}
  evals = [(x, True) for x in train_datasets] + [(x, False) for x in eval_datasets]
  for ds, is_train in evals:
    task = ds.dataset.get_task()
    if task == Task.CAPTIONING:
      metric_name, name = "cider", "cider"
      k = evaluator.ResultKey(metric_name="bleu4", dataset_name=ds.get_name())
      other_log[k] = "bleu4"
    elif task == Task.CLS:
      metric_name, name = "accuracy", "cls"
    elif task == Task.VQA:
      metric_name, name = "score", "vqa"
    elif task == Task.DETECTION:
      metric_name, name = "AP", "loc"
    elif task == Task.CLS_IN_CONTEXT:
      metric_name, name = "accuracy", "ident"
    elif task == Task.WEBQA:
      metric_name, name = "accuracy", "webqa"
    else:
      raise RuntimeError()
    name = f"train-evals/{name}" if is_train else f"val-evals/{name}"
    other_log[evaluator.ResultKey(metric_name=metric_name, dataset_name=ds.get_name())] = name
    qtypes = WebQaDataset.QTYPES_NAME_TO_TYPES.get("basic", ("basic",))
    # Set the val set for debugging since loading the train set is slow
    webqa_val = WebQaDataset("val",  None, qtypes)
    webqa_eval = EvaluationSetup(
      evaluator.WebQaEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 5),
           answer_options=webqa_train.get_answer_options(False))
    )

    # Add the WebQaDataset we want to experiment with
    # trainer.train_datasets = []
    # trainer.eval_datasets = []
    # trainer.best_model_key = [ResultKey(
    #   metric_name="accuracy", dataset_name="webqa-val")]

    train_datasets.append(TrainerDataset(
      webqa_train, "webqa-tr",
      train_sample=1.0,
      eval_sample=3000, eval_setup=webqa_eval
    ))
    eval_datasets.append(TrainerDataset(
      webqa_val, "webqa-val", eval_sample=12000, eval_setup=webqa_eval))
    #trainer.best_model_key.append(ResultKey("accuracy", dataset_name="webqa-val"))

  return train_datasets, eval_datasets, evaluation,other_log
@dataclass
class EvaluationSetup(FromParams):
  """Specifies how to evaluate a task"""

  @classmethod
  def from_params(
    cls,
    params: Params,
    constructor_to_call=None,
    constructor_to_inspect=None,
    **extras
  ):
    if "iterator" in params:
      assert params.pop("iterator") is None

    # Manually build the troublesome  `prediction_args` field becaues allennlp
    # does not handle the dictionary-of-union case
    if "beam_search" in params:
      bs = params.pop_bool("beam_search")
      prediction_args = dict(allennlp_spec=BeamSearchSpec(**bs))
    else:
      prediction_args = params.pop("prediction_args")
    params["prediction_args"] = None
    out = super().from_params(params, constructor_to_call, constructor_to_inspect)
    for k, v in prediction_args.items():
      if isinstance(v, (int, float, str) or v is None):
        pass
      else:
        prediction_args[k] = PredictionArg.from_params(v)
    out.prediction_args = prediction_args
    return out

  evaluator: Evaluator
  prediction_args: Dict[str, Union[PredictionArg, int, float, str, None]]


@dataclass
class TrainerDataset(FromParams):
  """Dataset with meta-data that will be used during training"""

  dataset: Dataset
  logging_name: str = None
  eval_sample: Optional[int] = None
  train_sample: Union[int, float, None] = None
  eval_setup: EvaluationSetup = None

  def get_name(self):
    if self.logging_name is None:
      return self.dataset.get_name()
    else:
      return self.logging_name


@dataclass
class RunArgs(FromParams):

  @classmethod
  def from_params(cls, params: Params, constructor_to_call=None, constructor_to_inspect=None, **other):
    if "send_model" in params:
      del params["send_model"]
    return super().from_params(params, constructor_to_call, constructor_to_inspect, **other)

  """Specifies what devices/distributed setting to train on"""
  devices: Union[str, int, List[int], None]
  seed: int
  dist_backend: str = "nccl"
  dist_url: str = 'tcp://localhost:10001'
  grad_accumulation: int = 1
  num_workers: Optional[int] = None

  @property
  def distributed(self):
    return isinstance(self.devices, list)

  @staticmethod
  def build(args: 'DeviceArgsType', force_one_process=False,
            grad_accumulation=1, num_workers=num_workers, seed=None, dist_port=None, dist_backend="nccl"):
    if isinstance(args, RunArgs):
      return args
    if args is None:
      if torch.cuda.is_available():
        logging.info("cuda found, defaulting to cuda")
        args = 'cuda'
      else:
        logging.info("cuda not found, using cpu")
        args = 'cpu'

    elif isinstance(args, list) and len(args) == 1 and not force_one_process:
      args = args[0]
    if seed is None:
      seed = np.random.randint(0, 2**28)

    if dist_port is not None:
      dist_port = f'tcp://localhost:{dist_port}'
    elif "PYTORCH_DIST_PORT" in os.environ:
      dist_port = f'tcp://localhost:{os.environ["PYTORCH_DIST_PORT"]}'
    else:
      dist_port = f'tcp://localhost:64801'
    return RunArgs(args, grad_accumulation=grad_accumulation,
                   num_workers=0, seed=seed, dist_url=dist_port)


# Arguements we can use to specify the device
DeviceArgsType = Union[RunArgs, int, str, List[int], None]


@dataclass
class _TrainingState:
  """Internal training state used for checkpointing"""
  global_step: int = 0
  epoch: int = 0
  best_save_score: Optional[float] = None
  loss_ema: float = 0.0
  monitor_ema: Optional[Dict] = None
  optimizer_state: Optional[Dict] = None
  scheduler_state: Optional[Dict] = None
  epoch_scheduler_state: Optional[Dict] = None
  model_state: Optional[Dict] = None



@dataclass
class _EvaluationRunner:
  """Internal class to run evaluations"""
  evaluator: Evaluator
  prediction_args: Dict[str, Any]
  examples: List
  data_loader: DataLoader
  dataset: Dataset
  eval_sample: int
  desc: str = "eval"
  nopbar: bool = False
  distributed_evaluator: bool = False

  def get_predictions(self, model) ->  Dict[str, GPVExampleOutput]:
    return run_model(
      model, self.data_loader, beams_to_keep=1,
      model_device=None, desc=self.desc, nopbar=self.nopbar,
      prediction_args=self.prediction_args
    )


def select_subdir(output_dir, target=None):
  prefix = "" if target is None else target + "-"
  i = 0
  while True:
    candidate = join(output_dir, prefix + "r" + str(i))
    if not exists(candidate):
      try:
        mkdir(candidate)
        return candidate
      except FileExistsError:
        pass
    i += 1


def is_distributed():
  return dist.is_initialized()


def is_primary():
  return not dist.is_initialized() or dist.get_rank() == 0


def get_lrs(optimizer):
  lrs = []
  for param_group in optimizer.param_groups:
    lrs.append(param_group['lr'])
  return lrs


def _train_worker(rank, *args, **kwargs):
  return Trainer._train_worker(*args, **kwargs, rank=rank)


def _remove_tensors(monitor: Dict) -> Dict:
  for k, v in monitor.items():
    if isinstance(v, torch.Tensor):
      monitor[k] = v.item()
  return monitor


@dataclass
class Trainer(FromParams):
  train_datasets: List[TrainerDataset]
  eval_datasets: List[TrainerDataset]
  evaluation: Dict[Task, EvaluationSetup]
  optimizer: OptimizerBuilder
  epochs: int
  train_loader: DataLoaderBuilder

  # Additional optimization settings
  step_schedule: TrainingScheduleBuilder = None
  clip_grad_norm_re: Optional[str] = None
  clip_grad_norm: Optional[float] = None

  #When to move onto next lesson 
  num_no_change_val: Optional[float] = None
  upper_bound_no_change: Optional[float] = None  
  # Data iterator parameters
  find_unused_parameters: bool = True
  end_at_epoch: Optional[int] = None
  eval_loader: DataLoaderBuilder = None
  output_dir: Optional[str] = None
  combine_lessons: Optional[str] = False
  combine_lesson_2: Optional[str] = False
  # Should we balance the different train dataset between batches
  stratify: bool = False

  lesson_training: bool = False
  # Saving
  save_evaluation_results: bool = True
  save_prediction_samples: Optional[int] = 0
  save_each_epoch: int = True
  best_model_key: Union[ResultKey, None, List[Union[ResultKey, Tuple[ResultKey, float]]]] = None
  eval_at_start: bool = False
  checkpoint: bool = False
  train_val_log: Optional[List[Tuple[ResultKey, str]]] = None
  best_val: int = None
  # Cosmetic/Logging
  tb_log_intervals: int = 20
  tb_log: bool = True
  log_lr: bool = True
  log_frozen_parameters = True
  loss_logging_ema: float = 0.99
  monitor_ema: float = 0.99
  eval_pbar: bool = True
  epoch_pbar: bool = True
  eval_at: int = None
  sync_monitor: bool = True
  combine_lesson_2:bool = False 
  train_state_file = None
  old_model = None
  actual_epoch = None
  epoch_scheduler = None  
  best_trajec_score = None 
  prefix = None 
  val_score = None 

  @classmethod
  def from_params(
      cls,
      params: Params,
      constructor_to_call = None,
      constructor_to_inspect = None,
      **extras
  ):
    if "sort_train" in params:
      # No longer supported
      assert not params.pop("sort_train")
    out: Trainer = super().from_params(params)

    # Default from params does not do str -> class
    # conversion for dictionary keys, so manually fix here for the `evaluation` field
    evaluation = {}
    for k, v in out.evaluation.items():
      if isinstance(k, str):
        evaluation[Task(k)] = v
      else:
        evaluation[k] = v
    out.evaluation = evaluation
    return out

  def train(self, model: GPVModel, output_dir: Optional[str],
            device: DeviceArgsType = None, override=False):
    if output_dir is not None:
      logging.info(f"Initializing model dir {output_dir}")
      #clear_if_nonempty(output_dir, override)
      makedirs(output_dir, exist_ok=True)
      Params(to_params(self, None)).to_file(join(output_dir, "trainer.json"))
      Params(to_params(model, GPVModel)).to_file(join(output_dir, "model.json"))
    else:
      logging.info(f"No output dir, model will not be saved")
    device = RunArgs.build(device)
    self._train(model, output_dir, device)

  @staticmethod
  def resume_from_checkpoint(run_dir: str, device: DeviceArgsType = None, save=True):
    logging.info(f"Resuming training for {run_dir}")

    run_dir = select_run_dir(run_dir)

    status = load_json_object(join(run_dir, "status.json"))
    if status["done"]:
      logging.info(f"{run_dir} is already marked as done")
      return

    logging.info("Loading trainer")
    output_dir = dirname(run_dir)
    with py_utils.DisableLogging():
      trainer = Trainer.from_params(Params.from_file(join(output_dir, "trainer.json")))
    model_file = join(output_dir, "model.json")
    loc_setup = EvaluationSetup(
      evaluator.LocalizationEvaluator(),
      dict(beam_search_spec=None)
    )
    trainer.train_datasets = []
    trainer.eval_datasets = []
    trainer.train_datasets.append(TrainerDataset(ImageContrastDataset('train'),"img-contrast"))
    trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=4857,eval_setup=loc_setup))
    trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
    #train_datasets, eval_datasets, evaluation,other_log = get_datasets()
    #trainer.train_datasets = train_datasets
    #trainer.eval_datasets = eval_datasets
    # trainer.evaluation = evaluation
    # trainer.train_val_log = list(other_log.items())
    # qtypes = "basic"
    # qtypes = WebQaDataset.QTYPES_NAME_TO_TYPES.get("basic",("basic",))
    # webqa_train = WebQaDataset("train",None,qtypes)
    # webqa_val = WebQaDataset("val",None,qtypes)
    # webqq_eval = EvaluationSetup(
    #      evaluator.WebQaEvaluator(),
    #      dict(beam_search_spec=BeamSearchSpec(1,5),answer_options=webqa_train.get_answer_options(False)))
    # #trainer.train_datasets.append(TrainerDataset(webqa_train,"webqa-tr",train_sample=0.2,eval_sample=3000,eval_setup=webqq_eval))
    # for i,e in enumerate(trainer.eval_datasets):
    #     print(e.logging_name,e.logging_name=='webqa-val')
    #     if e.logging_name == 'webqa-val':
    #       ind_remove=i
    #       print(ind_remove,'ind rmeove')
    #       e.eval_setup = webqq_eval
    # for i,e in enumerate(trainer.train_datasets):
    #     if e.logging_name == 'webqa-tr':
    #       print(trainer.train_datasets[i])
    #       ind_remove_2 = i
    #       e.eval_setup = webqq_eval
    #       print(e.eval_setup)
    # #trainer.eval_datasets.pop(ind_remove)
    # #trainer.train_datasets.pop(ind_remove_2)
    # #trainer.train_datasets.append(TrainerDataset(webqa_train,"webqa-tr",train_sample=0.2,eval_sample=3000,eval_setup=webqq_eval))
    # #trainer.eval_datasets.append(TrainerDataset(webqa_val,"webqa-val",12000,eval_setup=webqq_eval))
    # trainer.best_model_key.append(ResultKey("accuracy",dataset_name="webqa-val"))
    # print(len(trainer.train_datasets),trainer.train_datasets[-1])
    # print(len(trainer.eval_datasets),trainer.eval_datasets[0])
    checkpoint_file = join(run_dir, "checkpoint.pth")

    run_args = RunArgs.build(device)
    if not save:
      logging.info("Save is false, so no results will be recorded")
      run_dir, output_dir = None, None
    trainer._train(model_file, output_dir, run_args, checkpoint_file, run_dir)

  @staticmethod
  def train_another_model(output_dir: str, device: DeviceArgsType = None, save=True):
    logging.info(f"Starting another run for {output_dir}")
    logging.info("Getting trainer/model")
    with py_utils.DisableLogging():
      trainer = Trainer.from_params(Params.from_file(join(output_dir, "trainer.json")))
      model = GPVModel.from_params(Params.from_file(join(output_dir, "model.json")))

    run_args = RunArgs.build(device)
    if not save:
      logging.info("Save is false, so no results will be recorded")
      output_dir = None
    trainer._train(model, output_dir, run_args)

  def _init_eval(
      self,
      model: GPVModel,
      train_examples: List[List],
      eval_examples: List[List]
  ) -> Dict[str, _EvaluationRunner]:
    runners = {}
    collate_fn = CollateWithBatch(model.get_collate())

    to_eval = list(zip(eval_examples, self.eval_datasets))
    #print(eval_examples,'eval examples')
    #print(len(eval_examples),'to eval')
    #to_eval += list(zip(train_examples, self.train_datasets))

    # for t_e, t_d in zip(train_examples,self.train_datasets):
    #   if t_d.logging_name != 'img-contrast':
        
    #     to_eval += list(t_e,t_d)

    builder = self.eval_loader

    if builder is None:
      builder = self.train_loader

    batch_size = builder.batch_size

    if is_distributed():
      assert batch_size % dist.get_world_size() == 0
      batch_size = batch_size // dist.get_world_size()

    total_eval = 0
    for examples, ds in to_eval:
      if ds.eval_sample == 0:
        continue
      # Slightly more efficient to group by query length
      prepped = [model.preprocess_example(x) for x in examples]
      do_sort = any(ex.sort_len is not None for ex in prepped)
      if do_sort:
        prepped.sort(key=lambda ex: (ex.sort_len, ex.id), reverse=True)
      else:
        # Ensures order is consistent, needed in the distributed case
        prepped.sort(key=lambda ex: ex.id)

      if ds.eval_setup is None:
        eval_spec = self.evaluation[ds.dataset.get_task()]
      else:
        eval_spec = ds.eval_setup

      if is_distributed():
        sampler = DistributedSubsetSampler(
          len(prepped), ds.eval_sample, dist.get_rank(), dist.get_world_size(), do_sort)
      else:
        if ds.eval_sample:
          sampler = SubsetSampler(len(prepped), ds.eval_sample, do_sort)
        else:
          sampler = None

      loader = builder.build(
        prepped, collate_fn, shuffle=False, sampler=sampler, batch_size=batch_size)
      total_eval += ds.eval_sample if ds.eval_sample else len(examples)

      pbar = is_primary() and self.eval_pbar
      runner = _EvaluationRunner(
        eval_spec.evaluator, eval_spec.prediction_args, examples,
        eval_sample=ds.eval_sample, data_loader=loader, dataset=ds.dataset,
        desc=f"{ds.get_name()}", nopbar=not pbar,
        # TODO this is a hack
        distributed_evaluator=not isinstance(eval_spec.evaluator, CaptionEvaluator)
      )
      if ds.get_name() in runners:
        raise ValueError("Datasets have identical logging names")
      runners[ds.get_name()] = runner

    return runners

  def _get_optimizers(self, model: GPVModel, epoch_size: int, train_state: _TrainingState):
    optimizer = self.optimizer.build(model, epoch_size, self.epochs)
    if train_state.optimizer_state:
      optimizer.load_state_dict(train_state.optimizer_state)
      train_state.optimizer_state = None

    step_scheduler = None
    if self.step_schedule is not None:
      #num_steps = epoch_size * self.epochs
      #need better way other than hardcoding 
      num_steps = epoch_size *15
      step_scheduler = self.step_schedule.build(optimizer, num_steps, train_state.global_step - 1)
      if train_state.scheduler_state is not None:
        step_scheduler.load_state_dict(train_state.scheduler_state)
    epoch_scheduler = None 
    if self.epoch_scheduler is not None:
      #epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.95)
      epoch_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
      if train_state.epoch_scheduler_state is not None:
        epoch_scheduler.load_state_dict(train_state.epoch_scheduler_state)
        optimizer = epoch_scheduler.step(optimizer)
      new_lr = optimizer.param_groups[0]['lr']
      logging.info(f'Current learning rate:{new_lr}')
      logging.info(f'stored schedule {epoch_scheduler.last_best_val_score}')
      #print(optimizer.param_groups[0]['lr'])

    return optimizer, step_scheduler,epoch_scheduler

  def _get_train_loader(self, model: GPVModel, training_examples: List[List],
                        runtime: RunArgs):
    #modify load_and_log_train
    #for each training dataset add list of examples, train loader,sampler 
    id_sets = set()
    global_all_train = []
    all_train = []
    new_all_train = []
    total_lesson_datasets = []
    print(len(training_examples),'training examples')
    for lesson in training_examples:
      #print(lesson,'lesson')
      all_train.append(py_utils.flatten_list(model.preprocess_example_train(x) for x in lesson))
    all_train_sizes = [len(x) for x in all_train]
    assert len(new_all_train) <= 40
    total_num_examples = 0
    syn_examples = []
    image_contrast_examples = []
    text_contrast_examples = []
    mil_examples = []
    for lesson in all_train:

       if lesson[0].task == Task.SYNONYM:
        new_all_train = []
        for i in range(0,len(lesson),2):
        #   print(all_train[i].image_id == all_train[i+1].image_id)
          syn_examples.append([lesson[i],lesson[i+1]])
          new_all_train.append([lesson[i],lesson[i+1]])
          global_all_train.append([lesson[i],lesson[i+1]])
        if len(new_all_train) <4:
          raise TypeError
        total_num_examples += len(new_all_train)
        sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(new_all_train))),batch_size=4,drop_last=True)
        loader = self.train_loader.build(new_all_train, model.get_collate(True),batch_size=1, shuffle=False, batch_sampler=sampler)
        # try:
        #   if len(list(loader)) == 0:
        #     print('loader 0')
        #     pdb.set_trace()
        # except IndexError:
        #   pdb.set_trace()
        #pdb.set_trace()
        total_lesson_datasets.append(loader)
       elif lesson[0].task == Task.IMAGECONTRAST or lesson[0].task == Task.TEXTCONTRAST:
        new_all_train = []
        train_dict = {}
        new_train_dict = {}
        for t in lesson:
          if t.meta not in train_dict:
            train_dict[int(t.meta)] = []
          train_dict[int(t.meta)].append(t)
        for k in train_dict.keys():
          if len(train_dict[k]) > 1:
            new_train_dict[k] = train_dict[k]
        new_all_train = list(new_train_dict.values())
        last_all_train = []
        sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(new_all_train))),batch_size=1,drop_last=True)
        for i in new_all_train:
          if new_all_train[0][0].task == Task.IMAGECONTRAST:
              image_contrast_examples.append(i)
              global_all_train.append(i)
              last_all_train.append(i)
          if new_all_train[0][0].task == Task.TEXTCONTRAST:
              text_contrast_examples.append(i)
              global_all_train.append(i)
              last_all_train.append(i)
        if len(new_all_train) ==0:
          raise TypeError
        if len(last_all_train) ==0:
          raise TypeError
        total_num_examples+= len(new_all_train)
        loader = self.train_loader.build(last_all_train, model.get_collate(True),batch_size=1, shuffle=False, batch_sampler=sampler)
        # try:
        #   if len(list(loader)) == 0:
        #     print('loader 0')
        #     pdb.set_trace()
        # except IndexError:
        #   pdb.set_trace()
        total_lesson_datasets.append(loader)
       elif lesson[0].correct_answer != None:
            new_all_train = []
            for i in range(len(lesson)):
              assert lesson[i].task == Task.MIL
              mil_examples.append(lesson[i])
              new_all_train.append(lesson[i])
            global_all_train.append(new_all_train)
            # if len(new_all_train) <16:
            #   raise TypeError
           
            total_num_examples+= len(new_all_train)
            sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(new_all_train))),batch_size=16,drop_last=True)
            loader = self.train_loader.build(new_all_train, model.get_collate(True),batch_size=1, shuffle=False, batch_sampler=sampler)
            # try:
            #   if len(list(loader)) == 0:
            #     print('loader 0')
            #     pdb.set_trace()
            # except IndexError:
            #   print('loader 0')
            #   pdb.set_trace()
            total_lesson_datasets.append(loader)

    # if not self.lesson_training:
    #   for grp in training_examples:
    #     all_train.append(py_utils.flatten_list(model.preprocess_example_train(x) for x in grp))
    #   all_train_sizes = [len(x) for x in all_train]
    #   #all_train = py_utils.flatten_list(all_train)
    
    #   shuffle = True
    #   batch_size = self.train_loader.batch_size
    #   if (any(x.train_sample is not None for x in self.train_datasets) or
    #       self.stratify or is_distributed()):
    #     # Use our custom sampler that handles all these cases
    #     if is_distributed():
    #       world_size, rank = dist.get_world_size(), dist.get_rank()
    #       if batch_size % world_size != 0:
    #         raise ValueError("Batch size not divisible by world size")
    #       batch_size = batch_size // world_size
    #       logging.info(f"Using batch size {batch_size} since there "
    #               f"are {world_size} workers with base size of {self.train_loader.batch_size}")
    #     else:
    #       world_size, rank = None, None

    #     samples = [x.train_sample for x in self.train_datasets]
    #     sampler = StratifiedSubsetSampler(
    #     all_train_sizes, runtime.seed, self.stratify, samples, batch_size, rank, world_size)
    #   else:
    #     shuffle = False   # Sampler does shuffling
    #     loader_batch_size = 1  # Sampler does batching
    #   batch_groups = runtime.grad_accumulation
    #   if batch_groups > 1:
    #     if batch_size % batch_groups != 0:
    #       raise NotImplementedError("Batch size not divisible by grad accumulation steps")
    #   prev_batch_size = batch_size
    #   batch_size = batch_size // batch_groups
    #   logging.info(f"Accumulating total of {prev_batch_size} through {batch_groups} size {batch_size} batches")
    #   #print(all_train[0],all_train[1])
    #   # sampler = StratifiedSubsetSampler(
    #   #   all_train_sizes, runtime.seed, self.stratify, samples, batch_size, rank, world_size)
    #   # print(all_train[0].task,'task')
    #   # print(all_train[0].task == 'synonym')
    #   # print(all_train[0].task == Task.SYNONYM)
    #   # if type(all_train[0]) == list:
    #   #   all_train = new_all_train
    #   print(all_train[0].task)
    #   #pdb.set_trace()
    #   if all_train[0].task == Task.SYNONYM:
    #     new_all_train = []
    #     for i in range(0,len(all_train),2):
    #     #   print(all_train[i].image_id == all_train[i+1].image_id)
    #       new_all_train.append([all_train[i],all_train[i+1]])
    #     all_train = new_all_train
    #     sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(all_train))),batch_size=8,drop_last=True)
    #   elif all_train[0].task == Task.IMAGECONTRAST or all_train[0].task == Task.TEXTCONTRAST:
    #     new_all_train = []
    #     train_dict = {}
    #     for t in all_train:
    #       if t.meta not in train_dict:
    #         train_dict[int(t.meta)] = []
    #       train_dict[int(t.meta)].append(t)
    #     new_all_train = list(train_dict.values())

    #     # print(new_all_train[10][0].id,'before')
    #     # random.shuffle(new_all_train)
    #     # print(new_all_train[10][0].id,'after')
    #     # print(new_all_train[10][1].id,'after')
    #     #sampler = SynonymSampler(new_all_train)
    #     #all_train = new_all_train
    #     #sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(new_all_train),batch_size=3,drop_last=True)
    #   # if len(all_train[0]) == 2:
    #   #sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(all_train),batch_size=2,drop_last=True)

    #   elif all_train[0].task == Task.IMAGECONTRAST or all_train[0].task == Task.TEXTCONTRAST:
    #     new_all_train = []
    #     train_dict = {}
    #     for t in all_train:
    #       if t.meta not in train_dict:
    #         train_dict[int(t.meta)] = []
    #       train_dict[int(t.meta)].append(t)
    #     new_all_train = list(train_dict.values())



    #     # new_all_train = []
    #     # for i in range(0,len(all_train),16):
    #     #   # for j in all_train[i:i+16]:
    #     #   #   print(j.index_of_class,'idx class')
    #     #   new_all_train.append(all_train[i:i+15])
        
    #     #   # print(new_all_train)
    #     #   # print(new_all_train[0][0].index_of_class,'contrast group 1')
    #     #   # print(new_all_train[0][15].index_of_class,'contrast group 2')
    #     # for entry in new_all_train:
    #     #   idx = entry[0].index_of_class
    #     #   for v in entry:
    #     #     if int(v.index_of_class) != int(idx):
    #     #       print(idx,v.index_of_class)
    #     #       raise Error
    #     # print(len(new_all_train),len(all_train),'all train and new all train')
    #     all_train = []
    #     sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(new_all_train))),batch_size=1,drop_last=True)
    #     for i in new_all_train:
    #       if new_all_train[0][0].task == Task.IMAGECONTRAST:
    #         if len(i)>= 10:
    #           all_train.append(i)
    #       if new_all_train[0][0].task == Task.TEXTCONTRAST:
    #         if len(i)>=10:
    #           all_train.append(i)

    #         # if len(i) >16:
    #         #   print('larger than 16')
    #     #all_train = new_all_train
    #     # for ex in all_train:
    #     #   if ex.task != Task.IMAGECONTRAST or ex.task != Task.TEXTCONTRAST:
    #     #     raise ValueError
    #     print(len(all_train),'length of all train')
    #     sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(all_train))),batch_size=1,drop_last=True)
    #   elif all_train[0].correct_answer != None:
    #     sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(all_train),batch_size=32,drop_last=True)
    #   else:
    #     sampler = sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(all_train),batch_size=32,drop_last=True)
      
    #   shuffle = False   # Sampler does shuffling
    #   loader_batch_size = 1  # Sampler does batching
    # else:
    #   loader_batch_size = batch_size
    #   sampler = None

    # batch_groups = runtime.grad_accumulation
    # if batch_groups > 1:
    #   if batch_size % batch_groups != 0:
    #     raise NotImplementedError("Batch size not divisible by grad accumulation steps")
    #   prev_batch_size = batch_size
    #   batch_size = batch_size // batch_groups
    #   logging.info(f"Accumulating total of {prev_batch_size} through {batch_groups} size {batch_size} batches")
    # #can change number of workers for loader here 

    # loader = self.train_loader.build(
    #   all_train, model.get_collate(True),
    #   batch_size=loader_batch_size, shuffle=shuffle, batch_sampler=sampler)

    # if batch_groups == 1:
    #   return loader, len(loader), sampler
    # else:
    #   batch_group_generator = lazy_groups_of(loader, batch_groups)
    #   num_training_batches = math.ceil(len(loader) / batch_groups)
    #   return batch_group_generator, num_training_batches, sampler

    logging.info(f'Total number of examples {total_num_examples}')
  

    sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(global_all_train))),batch_size=2,drop_last=True)
    # samples = [x.train_sample for x in self.train_datasets]
    # sampler = StratifiedSubsetSampler(
    # all_train_sizes, runtime.seed, self.stratify, samples, 15, None, None)
    # shuffle = False   # Sampler does shuffling
    # loader_batch_size = 1  # Sampler does batching
    loader = self.train_loader.build(global_all_train, model.get_collate(True),batch_size=1, shuffle=False, batch_sampler=sampler)

    return loader, total_num_examples
   

  def _get_train_eval_dir(self, run_dir, epochs, step) -> Optional[str]:
    if run_dir is None or (not self.save_evaluation_results and self.save_prediction_samples == 0):
      return None
    if not exists(join(run_dir, "train-evals")):
      mkdir(join(run_dir, "train-evals"))
    out = join(run_dir, "train-evals", f"ep{epochs}-st{step}")
    if not exists(out):
      mkdir(out)
    return out

  def _get_task_eval_dir(self, eval_dir, eval_name: str):
    task_dir = join(eval_dir, eval_name)
    if not exists(task_dir):
      mkdir(task_dir)

    return task_dir

  def _run_eval(self, model, runners: Dict[str, _EvaluationRunner],
                global_step: int, seed: int, eval_dir) -> Dict[ResultKey, Number]:
    if is_distributed():
      all_results = self._run_eval_dist(model, runners, global_step, seed, eval_dir)
    else:
      all_results = {}
      for name, eval in runners.items():
        outputs = eval.get_predictions(model)
        results = eval.evaluator.evaluate(eval.examples, outputs, allow_partial=True)
        if self.save_prediction_samples != 0 and eval_dir is not None:
          to_save = py_utils.sample_dict(outputs, self.save_prediction_samples)
          save_gpv_output(to_save, self._get_task_eval_dir(eval_dir, name))

        for k, v in results.items():
          all_results[replace(k, dataset_name=name)] = v

    return all_results

  def _run_eval_dist(self, model, runners: Dict[str, _EvaluationRunner],
                     global_step, seed, eval_dir) -> Dict[ResultKey, Number]:
    # Need to give any samplers an identical seed that is different from seeds
    # used in previous evaluations
    for runner in runners.values():
      if runner.data_loader.sampler is not None:
        runner.data_loader.sampler.set_seed(global_step*78 + seed*13)

    n_to_save = self.save_prediction_samples

    all_results = {}
    predictions = {}
    for name, eval in runners.items():
      pred = eval.get_predictions(model)
      if eval_dir and n_to_save != 0:
        # Primary saves its predictions
        out = self._get_task_eval_dir(eval_dir, name)
        to_save = py_utils.sample_dict(pred, self.save_prediction_samples)
        save_gpv_output(to_save, out)

      if not eval.distributed_evaluator:
        # Each process sends over all its predictions
        predictions[name] = pred
      else:
        # Each process runs its own evaluation and just sends the results
        results = eval.evaluator.evaluate(
              eval.examples, pred, allow_partial=True, mean=False)
        for k, v in results.items():
          all_results[replace(k, dataset_name=name)] = v

    # Workers run the evaluations completely independently, now gather the results on the primary
    logging.info(f"Aggregating distributed results")

    if len(all_results) > 0:
      # Gather scores where the each process ran its own evaluator
      # We don't use `gather_object` since it is not supported on NCCL backend
      output = [None for _ in range(dist.get_world_size())]
      dist.all_gather_object(output, all_results)

      if is_primary():
        # Aggregate distributed results, which were reported in (total, count) form
        all_results = py_utils.transpose_list_of_dicts(output)
        for k, v in all_results.items():
          sums, ns = py_utils.transpose_lists(v)
          all_results[k] = sum(sums) / sum(ns)

    if len(predictions) > 0:
      # Gather scores where each process produced predictions, and the primary
      # runs the evaluator
      output = [None for _ in range(dist.get_world_size())]
      dist.all_gather_object(output, predictions)
      if is_primary():
        for evaluation_name in output[0]:
          all_pred = {}
          for pred in output:
            assert not any(x in all_pred for x in pred)
            all_pred.update(pred[evaluation_name])

          runner = runners[evaluation_name]
          results = runner.evaluator.evaluate(runner.examples, all_pred, True)
          for k, v in results.items():
            all_results[replace(k, dataset_name=evaluation_name)] = v

    if is_primary():
      return all_results
    else:
      return None

  def _log_results(
      self, result: Dict[ResultKey, Number], summary_writer: SummaryWriter, global_step,
      eval_time, eval_dir):
    if not is_primary():
      return

    if self.save_evaluation_results and eval_dir is not None:
      to_save = {str(k): v for k, v in result.items()}
      to_save["step"] = global_step
      with open(join(eval_dir, "eval.json"), "w") as fh:
        json.dump(to_save, fh, indent=2)
      save = {'val_score':result.values()}
      

    dataset_to_task = {}
    for k in self.train_datasets + self.eval_datasets:
      dataset_to_task[k.get_name()] = k.dataset.get_task()

    grouped_by_task = defaultdict(dict)
    for k, r in result.items():
      grouped_by_task[dataset_to_task[k.dataset_name]][k] = r

    result_str = f"Evaluation took {duration_to_str(eval_time)}, showing results"
    for task in Task:
      if task not in grouped_by_task:
        continue
      dict_of_dicts = defaultdict(dict)
      for k, r in grouped_by_task[task].items():
        if k.subset_name is None:
          key = k.metric_name
        else:
          key = k.subset_name + "-" + k.metric_name
        dict_of_dicts[k.dataset_name][key] = r
      result_str += "\n"
      result_str += py_utils.dict_of_dicts_as_table_str(dict_of_dicts, "%.3f", top_right=str(task))
    logging.info(result_str)

    if summary_writer:
      for k, r in result.items():
        if k.subset_name is None:
          key = k.metric_name
        else:
          key = k.subset_name + "-" + k.metric_name
        key = k.dataset_name + "/" + key
        summary_writer.add_scalar(key, r, global_step)

        if self.train_val_log:
          train_val_log = {k: v for k, v in self.train_val_log}
          if k not in train_val_log:
            continue
          summary_writer.add_scalar(train_val_log[k], r, global_step)

  def _init_run_dir(self, output_dir, run_args: RunArgs):
    """Initialize the subdirectory for this particular run"""

    run_dir = select_subdir(output_dir)
    logging.info("Saving run to %s" % run_dir)

    with open(join(run_dir, "runtime.json"), "w") as f:
      json.dump(dict(
        hostname=socket.gethostname(),
        date=datetime.now().strftime("%m%d-%H%M%S"),
        device=to_params(run_args, RunArgs)
      ), f, indent=2)
    dump_json_object(dict(done=False), join(run_dir, "status.json"))
    return run_dir

  def _log_continue(self, run_dir, run_args: RunArgs, train_state: _TrainingState):
    """Log a continue-from-checkpoint note to the runtime file"""

    runtime_f = join(run_dir, "runtime.json")
    prev = load_json_object(runtime_f)
    if "continue" not in prev:
      prev["continue"] = []
    prev["continue"].append(dict(
        global_step=train_state.global_step,
        epoch=train_state.epoch,
        hostname=socket.gethostname(),
        date=datetime.now().strftime("%m%d-%H%M%S"),
        device=to_params(run_args, RunArgs)
    ))
    with open(runtime_f, "w") as f:
      json.dump(prev, f, indent=2)

  def _get_model_score(self, results: Dict[ResultKey, Number]):
    """Get the overall model selection score from the computed results"""

    if isinstance(self.best_model_key, ResultKey):
      return results[self.best_model_key]

    total = 0
    for k in self.best_model_key:
      if isinstance(k, tuple):
        k, w = k
      else:
        w = 1.0
      total += results[k]*w
    return total

  def _load_and_log_train(self):
    """Load the training and log dataset sizes"""
    #create list of training examples 
  
    #print(self.train_datasets,len(self.train_datasets))
    training_examples = [x.dataset.load() for x in self.train_datasets]

    total = sum(len(x) for x in training_examples)

    logging.info(f"Have {total} train examples")
    for ex, ds in zip(training_examples, self.train_datasets):
      logging.info(f"\t{len(ex)} from {ds.get_name()}")
    return training_examples

  def _load_and_log_eval(self, training_examples):
    """Load eval data and log the dataset sizes"""
    eval_examples = [x.dataset.load() for x in self.eval_datasets]

    total_eval = 0
    all_eval = (list(zip(eval_examples, self.eval_datasets))) 
    #+list(zip(training_examples, self.train_datasets)))
    # all_eval = (list(zip(eval_examples, self.eval_datasets)) +
    #             list(zip(training_examples, self.train_datasets)))
    for (examples, ds) in all_eval:
      if ds.eval_sample:
        total_eval += ds.eval_sample
      else:
        total_eval += len(examples)

    logging.info(f"Have {total_eval} eval examples")
    for (examples, ds) in all_eval:
      if ds.eval_sample == 0:
        pass
      elif ds.eval_sample is not None:
        logging.info(f"\t{ds.eval_sample} samples of {len(examples)} from {ds.get_name()}")
      else:
        logging.info(f"\t{len(examples)} from {ds.get_name()}")
    return eval_examples

  def _train(self, model: Union[str, GPVModel], output_dir, runtime: RunArgs,
             train_state_file: Optional[str]=None, run_dir=None):
    """Train with the output dir already initialized, and possibly a checkpoint file"""

    if train_state_file is not None:
      assert isinstance(model, str)
    elif self.train_state_file is not None:
      train_state_file = self.train_state_file
      model = self.old_model
 
    else:
      assert isinstance(model, GPVModel)

    if not runtime.distributed:
      # Load train since we can pass it directly to the worker method
      logging.info("Loading training data")
      #need to modify 
      training_examples = self._load_and_log_train()
    else:
      # Only load train if we need to initialize the moel
      training_examples = None

    if output_dir is not None:
      if run_dir is None:
        logging.info("Initializing run dir")
        run_dir = self._init_run_dir(output_dir, runtime)
      log_file = join(run_dir, "out.log")
      record_log_handle = logging.FileHandler(log_file)
      logging.getLogger().addHandler(record_log_handle)
    else:
      run_dir = None
      record_log_handle = None

    devices = runtime.devices
    if isinstance(devices, list):
      world_size = len(devices)

      logging.info(f"Given {len(devices)} devices, launching {len(devices)} worker processes")
      if isinstance(model, str):
        to_send = model
      else:
        to_send = to_params(model, GPVModel)
      args = (
        self, to_send, None,
        run_dir, runtime, train_state_file, world_size
      )
      context = torch.multiprocessing.spawn(_train_worker, nprocs=world_size, args=args, join=False)

      del training_examples

      while not context.join():
        pass
    else:
      self._train_worker(model, training_examples, run_dir, runtime, train_state_file)

    if run_dir is not None:
      dump_json_object(dict(done=True), join(run_dir, "status.json"))

    if record_log_handle is not None:
      logging.getLogger().removeHandler(record_log_handle)

  def _train_worker(self, model, training_examples,
                    run_dir, runtime: RunArgs,
                    train_state_file: Optional[str], world_size=None, rank=0):
    """Train function that can be used in a distributed setting"""
    # Set the device, and do some setup if we are distributed
    if world_size is not None:

      # Need to reset the logging
      py_utils.add_stdout_logger()

      if rank == 0 and run_dir:
        log_file = join(run_dir, "primary.log")
        record_log_handle = logging.FileHandler(log_file)
        logging.getLogger().addHandler(record_log_handle)

      device = runtime.devices[rank]
      if rank == 0:
        suffix = " (will log to stdout)"
      else:
        suffix = ""
      logging.info(f"Worker {rank} proc={getpid()} starting for device {device}{suffix}")

      if not isinstance(model, GPVModel):
        # Make sure everything we need to build the model with from_params is imported
        our_utils.import_all()

      if rank > 0:
        logging.disable(logging.WARNING)  # Non-primary only logs critical messages
        run_dir = None  # Only the primary saves anything

      dist.init_process_group(
        backend=runtime.dist_backend,
        init_method=runtime.dist_url,
        world_size=world_size,
        rank=rank)
      torch.cuda.set_device(device)
    else:
      device = runtime.devices

    # Now get the train state and model as objects
    if train_state_file is not None:
      # resuming training
      logging.info("Loading train state")
      print(train_state_file,'train state file')
      train_state: _TrainingState = torch.load(train_state_file, map_location="cpu")
      #train_state: _TrainingState = load_json_object(train_state_file)
      # model is passed in as a file in this case
      logging.info("Loading model")
      with py_utils.DisableLogging():
        model = GPVModel.from_params(Params.from_file(model))
      model.load_state_dict(train_state.model_state)
      train_state.model_state = None
      train_state.epoch = 0 

      if run_dir is not None:
        self._log_continue(run_dir, runtime, train_state)
    else:
      train_state = _TrainingState()

      # Get the model
      if isinstance(model, dict):
        logging.info("Loading model")
        # Models was sent as parameters
        with py_utils.DisableLogging():
          model = GPVModel.from_params(Params(model))
      else:
        # Should have been sent the full model
        assert isinstance(model, GPVModel)

      if is_primary():
        logging.info("Initializing model")
        model.initialize()
      else:
        model.initialize(False)

    if train_state.epoch != 0:
      assert train_state.global_step != 0
   

    device = torch.device(device)

    # Finish setting up the model
    model.to(device)

    _base_model = model
    if world_size is not None:
      model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=self.find_unused_parameters)

    logging.info("Preparing training loader")
    if training_examples is None:
      training_examples = self._load_and_log_train()
    #print(training_examples,'training examples')

    train_loader,total_examples = self._get_train_loader(_base_model, training_examples, runtime)
    logging.info("Preparing evaluation")
    eval_examples = self._load_and_log_eval(training_examples)
    eval_runners = self._init_eval(_base_model, training_examples, eval_examples)

    logging.info("Preparing optimizers")
    optimizer, step_scheduler, epoch_scheduler = self._get_optimizers(
      _base_model, total_examples, train_state)
    if self.clip_grad_norm_re and self.clip_grad_norm is not None:
      clip_params = [p for n, p in _base_model.named_parameters() if re.match(self.clip_grad_norm_re, n)]
      logging.info(f"Clipping grad norm for {len(clip_params)} parameters")
    else:
      clip_params = list(model.parameters())

    # Other stuff we need to track during training
    if run_dir and self.tb_log:
      summary_writer = SummaryWriter(join(run_dir, "log"))
    else:
      summary_writer = None
    best_saved_score = train_state.best_save_score
    monitor_ema = train_state.monitor_ema
    global_step = train_state.global_step
    if monitor_ema is None:
      monitor_ema = {}
    loss_ema = train_state.loss_ema

    # Do initial eval if asked
    if self.eval_at_start and global_step == 0:
      logging.info("Starting initial eval")
      start = perf_counter()
      eval_dir = self._get_train_eval_dir(run_dir, 0, global_step)
      results = self._run_eval(_base_model, eval_runners, global_step, runtime.seed, eval_dir)
      self._log_results(results, summary_writer, global_step, perf_counter() - start, eval_dir)
      if is_primary() and self.best_model_key:
        best_saved_score = self._get_model_score(results)

    # Ready to start!
    if global_step > 0:
      logging.info(f"Resume training from ep={train_state.epoch} global_step={global_step}")
    else:
      logging.info(f"Start training")

    n_train = sum(p.requires_grad for p in model.parameters())
    n_freeze = sum(not p.requires_grad for p in model.parameters())
    logging.info(f"Have {n_train} params and {n_freeze} frozen parameters")
    if self.actual_epoch != 0:
      train_state.global_step = len(train_loader)*self.actual_epoch
    epoch_scheduler.last_epoch = self.actual_epoch
    for epoch in range(train_state.epoch, self.epochs):
      print(self.combine_lesson_2,'combine lesson 2')
      print(self.combine_lessons,'combine lessons ')
      total_l = 0
      ep_start = perf_counter() 
      model.train()
      pbar = tqdm(train_loader,disable=not self.epoch_pbar,ncols=100,desc='loss=',total=len(train_loader))
      for i,batch in enumerate(pbar):

          batch = our_utils.to_device(batch, device)
          
         
         

          loss, monitor = model(**batch)
          
          monitor = _remove_tensors(monitor)
       
          loss.backward()
          loss = loss.item()
          total_l += loss 
          # for group in optimizer.param_groups:
          #   for p in group['params']:
          #     print(p.grad)
        
          if i!= 0 and i%250 == 0:
            logging.info(f'Loss is {total_l/i} for batch {i} out of {len(pbar)}')
            summary_writer.add_scalar("250_loss",total_l/i,global_step)
          if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(clip_params, self.clip_grad_norm)

          optimizer.step()
          if not np.isfinite(loss):
            raise ValueError(f"non-finite foss {loss}")

          # Manually remove gradients, slightly faster then `optimizer.zero_grad`
          for group in optimizer.param_groups:
            for p in group['params']:
              p.grad = None

          global_step += 1
          if step_scheduler is not None:
            step_scheduler.step()

          if is_distributed() and self.sync_monitor:
            # Gather `monitor` from each work to primary so we can log the global average
            # We use `all_gather_object` so things work even if monitor had different
            # keys on different processes
            out = [None] * world_size
            dist.all_gather_object(out, (loss, monitor))
            if is_primary():
              loss = np.mean([x[0] for x in out])
              monitor = py_utils.transpose_list_of_dicts([x[1] for x in out])
              monitor = {k: np.mean(v) for k, v in monitor.items()}

          if is_primary():
            for k, v in monitor.items():
              if k not in monitor_ema:
                monitor_ema[k] = v
                to_show = v
              else:
                cur = monitor_ema[k]
                ema = cur * self.monitor_ema + v * (1 - self.monitor_ema)
                monitor_ema[k] = ema
                to_show = (ema / (1 - self.monitor_ema ** global_step))

              if summary_writer is not None and global_step % self.tb_log_intervals == 0:
                summary_writer.add_scalar(f"train/{k}", to_show, global_step)

            loss_ema = loss_ema * self.loss_logging_ema + loss * (1 - self.loss_logging_ema)
            corrected_loss_ema = (loss_ema / (1 - self.loss_logging_ema ** global_step))
            pbar.set_description("loss=%.4f" % corrected_loss_ema, refresh=False)

            if summary_writer is not None and global_step % self.tb_log_intervals == 0:
              summary_writer.add_scalar("train/loss-smoothed", corrected_loss_ema, global_step)
              summary_writer.add_scalar("train/loss", loss, global_step)

              if self.log_lr:
                for j, group in enumerate(optimizer.param_groups):
                  name = group.get("name", f"group_{j}")
                  summary_writer.add_scalar(f'lr/{name}', group["lr"], global_step)

              if self.log_frozen_parameters:
                for j, group in enumerate(optimizer.param_groups):
                  name = group.get("name", f"group_{j}")
                  n_frozen = sum(not x.requires_grad for x in group["params"]) / len(group["params"])
                  if n_frozen > 0:
                    summary_writer.add_scalar(f'lr/{name}-frozen', n_frozen, global_step)
      #print(epoch_scheduler.get_last_lr())
      #summary_writer.add_scalar('lr_epoch',epoch_scheduler.get_last_lr()[0],self.actual_epoch)
      ep_end = perf_counter()
      if self.eval_at is not None and (epoch+1) % self.eval_at != 0:
        continue

      logging.info(f"Epoch {self.actual_epoch + 1} took {duration_to_str(ep_end - ep_start)}, starting evaluation")

      eval_start = perf_counter()
      # Just eval the base model since we don't need any synchronization between models
      eval_dir = self._get_train_eval_dir(run_dir, epoch, global_step)
      results = self._run_eval(_base_model, eval_runners, global_step, runtime.seed, eval_dir)
      eval_end = perf_counter()
      val_score = list(results.values())[0]
      self.val_score = val_score 
      if self.best_val == None:
        self.best_val = int(val_score)
      else:
        if val_score > self.best_val:
          self.best_val = val_score
        else:
          self.num_no_change_val += 1
      logging.info(f"{self.upper_bound_no_change-self.num_no_change_val} epochs remaining")
      self._log_results(results, summary_writer, global_step, eval_end-eval_start, eval_dir)

      if summary_writer:
        summary_writer.add_scalar("time/train", ep_end-ep_start, epoch+1)
        summary_writer.add_scalar("time/eval", eval_end - eval_start, epoch + 1)

      if self.best_model_key and is_primary():
        score = self._get_model_score(results)
        score_dict = {"val":list(results.values())[0]}
        self.val_score = list(results.values())[0]
        dump_json_object(score_dict,run_dir+'/val_score.json')

          
        # if best_saved_score is None or best_saved_score < score:
        #   prefix = "Saving as" if run_dir else "Found"
        #   if best_saved_score is None:
        #     logging.info(f"{prefix} best model ({score:.5f}) ep={epoch+1}")
        #   else:
        #     logging.info(f"{prefix} best model ({score:.5f} > {best_saved_score:.5f}) ep={epoch+1}")
        #   best_saved_score = score
        #   if run_dir:
        #     best_model_file = join(run_dir, file_paths.BEST_STATE_NAME)
        #     torch.save(_base_model.state_dict(), best_model_file)
      # if not self.best_model_key:
      #   self.num_no_change_val += 1
    

      # if (run_dir is not None and
      #     self.save_each_epoch and
      #     epoch % self.save_each_epoch == 1):
      #   state_file = join(run_dir, f"state-ep{epoch+1}.pth")
      #   logging.info(f"Saving state as {state_file}")
      #   torch.save(_base_model.state_dict(), state_file)
      logging.info(f'best trajec score {self.best_trajec_score}')
      logging.info(f'current val score {self.val_score}')
      if run_dir is not None and self.checkpoint:
        if float(self.val_score) > float(self.best_trajec_score):
          if not os.path.exists(self.prefix+'/best_model/'):
            os.mkdir(self.prefix+'/best_model/')
          epoch_scheduler.last_best_val_score = self.val_score 
          Params(to_params(self, None)).to_file(join(self.prefix+'/best_model/', "trainer.json"))
          Params(to_params(model, GPVModel)).to_file(join(self.prefix+'/best_model/', "model.json"))
          checkpoint = _TrainingState(
              epoch=epoch+1,  # +1 because the epoch has ended
              loss_ema=loss_ema,
              global_step=global_step,
              monitor_ema=monitor_ema,
              scheduler_state=None if step_scheduler is None else step_scheduler.state_dict(),
              optimizer_state=optimizer.state_dict(),
              best_save_score=best_saved_score,
              model_state=_base_model.state_dict(),
              epoch_scheduler_state= epoch_scheduler.state_dict()
          )
          self.best_trajec_score = self.val_score 
          checkpoint_file = join(self.prefix+'/best_model/' "checkpoint.pth")
          logging.info(f"Checkpointing to {checkpoint_file}")
          logging.info(f'Update best trajec score to {self.val_score}')
          torch.save(checkpoint, checkpoint_file)

      if epoch == self.end_at_epoch:
        logging.info(f"Hit epoch {self.end_at_epoch}, ending early")
        break
      # if self.num_no_change_val >= self.upper_bound_no_change:
      #   logging.info(f"Validation score has decreased or not changed for too many epochs. Ending training")
      #   logging.info(f"Best score is {best_saved_score}")
      #   break
