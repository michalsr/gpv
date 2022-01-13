import json
import logging
import os
import sys 
from copy import deepcopy
from exp.ours.data.dataset import Task, GPV1_TASKS, GPV2_TASKS
from typing import List, Optional, Dict, Any, Union, Tuple
from exp.ours import params 
from utils import io
from torch.utils.tensorboard import SummaryWriter
import torch 
from exp.ours.train.trainer import TrainerDataset, RunArgs, Trainer, EvaluationSetup
import torch.utils.data
from transformers import AutoConfig
from allennlp.common import FromParams, Params, Registrable
from exp.ours.data.opensce import OpenSceDataset
from exp.ours.data.webqa import WebQaDataset, WebQaNoTemmplatesDataset
from exp.ours.data.image_contrast import ImageContrastDataset
from exp.ours.data.text_contrast import TextContrastDataset
from exp.ours.data.synonym import SynonymDataset 
from exp.ours.data.mil import MILDataset
from exp.ours.util.py_utils import clear_if_nonempty
from exp.ours.data.webqa_templates import WebQaQueryGenerator, TemplateWebQueryGenerator
from exp.ours.experiments.visual_model_cli import add_image_featurizer_args, get_image_featurizer
from exp.ours.models.layers import *
from exp.ours.models.losses import *
from exp.ours.models.t5_gpv import T5GPV
from exp.ours.models.t5_gpv_per_box import T5GpvPerBox
from exp.ours.models.t5_gpv import T5GPV
from exp.ours.train.evaluator import ResultKey
from exp.ours.train.optimizer_builder import AllParameters, OptimizerBuilder, \
  DelayedWarmupScheduleBuilder, ParameterGroup, AdamWBuilder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from exp.ours.experiments.trainer_cli import *
from exp.ours.data.dataset import Task
from exp.ours.util import py_utils

def get_model_and_trainer(args):
  if args.model is None:
    if args.debug in ["tiny", "small"] and args.init_from is None:
      args.model = "t5-small"
    else:
      args.model = "t5-base"

  image_featurizer, image_dim = get_image_featurizer(args)

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model
  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels'])

  model = T5GpvPerBox(
    args.model,
    loss=BasicGPVLoss(localization_loss),
    image_feature_extractor=image_featurizer,
    image_joiner=Linear(image_dim, t5_dim),
    pre_tokenize=True,
    query_box=None if args.query_box == "none" else args.query_box,
    all_lower_case=True,
    webqa_templates=TemplateWebQueryGenerator(use_commands=True),
    initialize_from=args.init_from,
    contrast_query="other",
    convert_to_relevance="raw",
    combine_with_objectness="multiply",
    embed_objectness_score=False
  )


  groups = [ParameterGroup(
    AllParameters(),
    group_name="other",
    overrides=dict(delay=0.0, warmup=0.1, lr=args.lr),
    allow_overlap=True
  )]

  scheduler = DelayedWarmupScheduleBuilder()
  optimizer = AdamWBuilder(
    lr=args.lr,
    weight_decay=args.weight_decay,
    parameter_groups=groups
  )

  print("Optimizer:")
  print(json.dumps(to_params(optimizer, OptimizerBuilder), indent=2))

  trainer: Trainer = get_trainer_from_args(
    args, logging_ema=0.995, find_unused_parameters=True,
    optimizer=optimizer, scheduler=scheduler)
    
  return model, trainer 


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
class Weights(nn.Module):
  def __init__(self,num_lessons) -> None:
      super().__init__()
      self.weights = nn.Parameter(torch.ones(num_lessons)*1/num_lessons)
      #self.weights.fill_(1/num_lessons)
      self.params = [self.weights] 
      self.optimizer = torch.optim.Adam(self.params)
      self.softmax = nn.Softmax(dim=None)
      self.log_prob = []
  def sample(self,trajec):

    dist = torch.distributions.categorical.Categorical(probs=self.softmax(self.weights))
    lesson = dist.sample()

    self.log_prob.append(dist.log_prob(lesson))
    #print(trajec,len(self.log_prob))
    assert len(self.log_prob) == trajec + 1
    #print(self.log_prob[f'trajec_{j}'].requires_grad)
    return lesson     
  def forward(self):
    return self.softmax(self.weights)
  def update(self,normalized_reward):
    tensor_reward = []
    loss_values = []
    for r in normalized_reward:
      value = r.split('_')
      #print(value[1])
      #type(int(float(value[1])))
      tensor_reward.insert(int(value[1]),normalized_reward[r])
    tensor_reward = torch.tensor(tensor_reward)

    for log_prob, reward  in zip(self.log_prob,tensor_reward):
      #print('hello I am here')
     
      #print(reward.requires_grad)
      #print(log_prob.requires_grad,'do i require grad here')
      r = -log_prob*reward

      loss_values.append(r)
    for v in loss_values:
      print(type(v),v.requires_grad)
    self.optimizer.zero_grad()
    loss_values = torch.stack(loss_values)
    loss = loss_values.mean()
    self.weights.grad = torch.autograd.grad(loss,self.parameters())[0]
    print(self.weights.grad)
    self.optimizer.step()




def run_trainer_from_args(trainer, model, args,output_dir):
  devices = RunArgs.build(get_devices(args.device), args.force_one_worker, args.grad_accumulation)
  trainer.train(model, output_dir, devices, override=args.override)
@dataclass
class AutoTask(FromParams):
  train_tasks: Optional[List[str]] = None
  gpv_model: Optional[T5GpvPerBox] = None 
  args: Optional[str] = None 
  policy_network: Optional[nn.Module] = None

  policy_opt: Optional[nn.Module] = None 
  lessons: Optional[List[str]] = None 
  trainer: Optional[Trainer] = None 
  num_trajec: Optional[int] = None 
  sampled_lesson: Optional[int] = None
  current_lesson_trajec: Optional[List] = None  
  map_int_to_lesson: Optional[dict] =  None 
  map_lesson_to_int: Optional[dict] = None 
  best_model_path: Optional[str] = None 
  trajec_to_validation_scores: Optional[dict] = None 
  trajec_to_normalized_scores:Optional[dict] = None
  trajec_to_output_dir: Optional[dict] = None
  epochs: Optional[int] = None
  log_prob: Optional[dict] = None 
  auto_logger: Optional[logging.Logger] = None 
  outer_log_step: Optional[int] = None
  inner_log_step: Optional[int] = None
  summary_writer: Optional[SummaryWriter] = None
 

  lesson_datasets = {'image_contrast':TrainerDataset(ImageContrastDataset('train'),"img-contrast"), "text_contrast":TrainerDataset(TextContrastDataset("train"),"text-contrast"),"mil":TrainerDataset(MILDataset('train'),'mil'),
    'synonym':TrainerDataset(SynonymDataset("train"),"synonym-train")}
  def reset(self):
      #reset trainer and model 
      model, trainer = get_model_and_trainer(self.args)
      self.gpv_model = model 
      self.trainer = trainer

  def initialize(self):
    #TODO allow for multiple train tasks and check that lessons match tasks
    #py_utils.add_stdout_logger()


    self.auto_logger =  setup_logger('auto_logger', params.GLOBAL_OUTPUT_FILE+'/log_file.log')
    self.map_int_to_lesson = {}
    self.map_lesson_to_int = {}

    self.summary_writer = SummaryWriter(params.GLOBAL_OUTPUT_FILE+'/tensorboard')
    for i,l in enumerate(self.lessons):
      self.map_int_to_lesson[i] = l
      self.map_lesson_to_int[l] = i
    self.inner_log_step = 0
    self.outer_log_step = 0
  def adjust_trainer(self,new_output_dir,init_from):
    self.trainer.train_datasets = []
    self.trainer.eval_datasets = []
    self.trainer.train_datasets.append(self.lesson_datasets[self.map_int_to_lesson[self.sampled_lesson.item()]])
    #TODO add evaluation for other tasks 
    loc_setup = EvaluationSetup(
          evaluator.LocalizationEvaluator(),
          dict(beam_search_spec=None)
            )
    #TODO add param for file name
    val_samples = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_10/val.json')
    self.trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=len(val_samples),eval_setup=loc_setup))
    self.trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
    self.trainer.stratify = True
    self.trainer.eval_loader = deepcopy(self.trainer.train_loader)
    self.trainer.train_loader.persist_workers = False
    self.trainer.eval_loader.persist_workers = False
    self.trainer.epochs = 1
    self.trainer.output_dir = new_output_dir
    self.gpv_model.initialize_from = init_from
    self.trainer.upper_bound_no_change = 100 
    self.trainer.num_no_change_val = 0
    self.auto_logger.info("Modified trainer for next lesson")
  def compute_normalized_validation(self):
    trajec_scores = list(self.trajec_to_validation_scores.values())
    mean = np.mean(trajec_scores)
    for i in self.trajec_to_validation_scores:
      self.trajec_to_normalized_scores[i] = self.trajec_to_validation_scores[i] - mean 
  def log_inner(self,trajec_num):
      key_value = f'trajec_{trajec_num}'
      val = self.trajec_to_validation_scores[key_value]
      self.summary_writer.add_scalar('trajectory_validation_scores',val,self.inner_log_step)
      self.inner_log_step += 1
  
  def log_outer(self,epoch):
      #record how many times each lesson gets selected
      #record validation and log prob scores for each lesson
      lesson_freq = {}
      weights = {}
      for l in self.lessons:
        lesson_freq[l] = 0
        weights[l] = 0
      for l in self.current_lesson_trajec:
        lesson_freq[l] += 1
      for i,v in enumerate(self.policy_network.weights.size()):
        weights[self.map_int_to_lesson[i]] = v
      self.summary_writer.add_scalars('lesson_freq',lesson_freq,epoch)
      self.summary_writer.add_scalars('weights',weights,epoch)
           
  def run(self):
    for i,e in enumerate(range(self.epochs)):
      if i ==0:
        self.initialize()
        self.auto_logger.info("Initialization complete")
      self.auto_logger.info(f'Epoch:{i}')
      for j in range(self.num_trajec):
        self.auto_logger.info(f'Trajectory:{j}')

        if j == 0:
            best_trajec_score = 49.67
            self.current_lesson_trajec = []
            self.trajec_to_output_dir = {}
            self.trajec_to_validation_scores = {}
            self.trajec_to_normalized_scores = {}
            self.log_prob = {}
          #sample 
        self.sampled_lesson = self.policy_network.sample(j)
        self.auto_logger.info(f"Sampled lesson {self.map_int_to_lesson[self.sampled_lesson.item()]} ")
        self.current_lesson_trajec.append(self.map_int_to_lesson[self.sampled_lesson.item()])
          #adjust trainer
        new_output_dir = f'{params.GLOBAL_OUTPUT_FILE}/epoch_{i}_lesson_{j}/'
        self.trajec_to_output_dir[f'trajec_{j}'] = new_output_dir
        if i == 0 or self.best_model_path == None:
          init_from = '/shared/rsaas/michal5/gpv_michal/outputs/seen_60_only_gpv_per_box/r0/best-state.pth'
        else:
          init_from = self.best_model_path 
        self.adjust_trainer(new_output_dir,init_from)
        run_trainer_from_args(self.trainer,self.gpv_model,self.args,new_output_dir)
        trajec_score = io.load_json_object(new_output_dir+'/r0/val_score.json')
        self.auto_logger.info(f"Trajectory {j} has reward {trajec_score['val']}")
  
        self.trajec_to_validation_scores[f'trajec_{j}'] = int(trajec_score['val'])
        if int(trajec_score['val']) > best_trajec_score:
          self.auto_logger.info(f"Best trajec score updated to {trajec_score['val']}")
          best_trajec_score = int(trajec_score['val'])
          self.best_model_path = self.trajec_to_output_dir[f'trajec_{j}'] 
        self.log_inner(j)
        self.reset()
      self.log_outer(i)
      self.compute_normalized_validation()
      self.policy_network.update(self.trajec_to_normalized_scores)
      self.auto_logger.info("Updated lesson distribution")
     

        

def main():
  parser = ArgumentParser()
  parser.add_argument("--model", choices=["t5-small", "t5-base", "t5-large"], default=None)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--webqa_sample", type=float, default=1.0)
  parser.add_argument("--webqa_subset",  default=None)
  parser.add_argument("--query_box",  default="always")
  parser.add_argument("--find_unused", action="store_true")
  parser.add_argument("--init_from")
  parser.add_argument("--train_from")
  parser.add_argument("--vwarmup", type=float, default=0.1)
  parser.add_argument("--sce", action="store_true")
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--vlr", type=float)
  parser.add_argument("--delay", type=float, default=0.0)
  parser.add_argument("--image_contrast",type=str,default=None)
  parser.add_argument("--text_contrast",type=str,default=None)
  parser.add_argument("--lesson",type=str,default=None)
  parser.add_argument("--mil",type=str,default=None)
  parser.add_argument("--synonym",type=str,default=None)
 
  add_image_featurizer_args(parser, vfreeze="all", vmodel="vinvl")
  add_train_args(
    parser, tasks=[str(Task.CAPTIONING)], epochs=4,
    clip_grad_norm=None, num_workers=4, batch_size=60)
  args = parser.parse_args()

  py_utils.add_stdout_logger()
  groups = [ParameterGroup(
    AllParameters(),
    group_name="other",
    overrides=dict(delay=0.0, warmup=0.1, lr=args.lr),
    allow_overlap=True
  )]

  scheduler = DelayedWarmupScheduleBuilder()
  optimizer = AdamWBuilder(
    lr=args.lr,
    weight_decay=args.weight_decay,
    parameter_groups=groups
  )
  clear_if_nonempty(params.GLOBAL_OUTPUT_FILE, override=False)
  os.makedirs(params.GLOBAL_OUTPUT_FILE, exist_ok=True)
  os.mkdir(params.GLOBAL_OUTPUT_FILE+'/tensorboard')
  model, trainer = get_model_and_trainer(args)
  a = AutoTask()
  a.train_tasks = params.TRAIN_TASKS
  a.gpv_model = model 
  a.args = args 
  a.policy_network = Weights(len(params.DET_LESSONS))

  #a.policy_network = SLP(len(params.DET_LESSONS))
  #a.policy_network.apply(init_weights)
  #a.policy_opt = torch.optim.Adam(a.policy_network.parameters())
  #print(len(list(a.policy_network.parameters())))
  a.lessons = params.DET_LESSONS
  a.trainer = trainer 
  a.num_trajec = 4
  a.epochs = 10
  a.run()

if __name__ == '__main__':
  main()
