import json
import logging
import os
from copy import deepcopy
from exp.ours.data.dataset import Task, GPV1_TASKS, GPV2_TASKS

import torch.utils.data
from transformers import AutoConfig

from exp.ours.data.opensce import OpenSceDataset
from exp.ours.data.webqa import WebQaDataset, WebQaNoTemmplatesDataset
from exp.ours.data.image_contrast import ImageContrastDataset
from exp.ours.data.text_contrast import TextContrastDataset
from exp.ours.data.synonym import SynonymDataset 
from exp.ours.data.mil import MILDataset
from exp.ours.data.vqa_classify_obj import VQA_CLS_OBJ_Dataset
from exp.ours.data.vqa_action_with_obj import VQA_ACT_W_OBJ_Dataset
from exp.ours.data.vqa_adj_with_obj import VQA_ADJ_W_OBJ_Dataset
from exp.ours.data.vqa_act_no_obj import VQA_ACT_NO_OBJ_Dataset
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
  parser.add_argument("--vqa_cls_obj",type=str,default=None)
  parser.add_argument("--vqa_act_w_obj",type=str,default=None)
  parser.add_argument("--vqa_adj_w_obj",type=str,default=None)
  parser.add_argument("--vqa_act_no_obj",type=str,default=None)
  add_image_featurizer_args(parser, vfreeze="all", vmodel="vinvl")
  add_train_args(
    parser, tasks=[str(Task.CAPTIONING)], epochs=4,
    clip_grad_norm=None, num_workers=4, batch_size=60)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.model is None:
    if args.debug in ["tiny", "small"] and args.init_from is None:
      args.model = "t5-small"
    else:
      args.model = "t5-base"

  image_featurizer, image_dim = get_image_featurizer(args)

  conf = AutoConfig.from_pretrained(args.model)
  t5_dim = conf.d_model
  localization_loss = DetrLocalizationLoss(1, 5, 2, 1, 0.5, 1, 5, 2, ['labels'])
  # model = T5GPV(
  #   args.model,
  #   loss=BasicGPVLoss(localization_loss),
  #   image_feature_extractor=image_featurizer,
  #   image_joiner=Linear(image_dim, t5_dim),
  #   pre_tokenize=True,
  #   image_relevance=SumWithObjectness(t5_dim, objectness_factor=True),
  #   query_box="always",
  #   all_lower_case=True,
  #   webqa_templates=TemplateWebQueryGenerator(use_commands=True),
  #   initialize_from=args.init_from
  # )

  # model = T5GPV(
  #   args.model,
  #   loss=BasicGPVLoss(localization_loss),
  #   image_feature_extractor=image_featurizer,
  #   image_joiner=Linear(image_dim, t5_dim),
  #   pre_tokenize=True,
  #   image_relevance=SumWithObjectness(t5_dim, objectness_factor=True),
  #   query_box=None if args.query_box == "none" else args.query_box,
  #   all_lower_case=True,
  #   webqa_templates=TemplateWebQueryGenerator(use_commands=True),
  #   initialize_from=args.init_from,
  # )
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
    optimizer=optimizer, scheduler=scheduler
  )
  trainer.output_dir = args.output_dir

  if args.webqa_subset is not None:

    if args.webqa_subset == "notemplates-5/6":
      qtypes = tuple("5a 5n 5v 6a 6v".split())
      webqa_train = WebQaNoTemmplatesDataset("train", 100 if args.debug else None, qtypes)
      webqa_val = WebQaNoTemmplatesDataset("val", 100 if args.debug else None, qtypes)
    elif args.webqa_subset == "notemplates-3/4":
      qtypes = tuple("3a 3n 3v 4a 4v".split())
      webqa_train = WebQaNoTemmplatesDataset("train", 100 if args.debug else None, qtypes)
      webqa_val = WebQaNoTemmplatesDataset("val", 100 if args.debug else None, qtypes)
    else:
      print('hi')
      qtypes = args.webqa_subset
      qtypes = WebQaDataset.QTYPES_NAME_TO_TYPES.get(qtypes, (qtypes,))
      webqa_train = WebQaDataset("val" if args.debug else "train",
                                 100 if args.debug else None, qtypes)
      webqa_val = WebQaDataset("val", 100 if args.debug else None, qtypes)

    webqq_eval = EvaluationSetup(
      evaluator.WebQaEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 5))
    )

    trainer.train_datasets.append(TrainerDataset(
      webqa_train, "webqa-tr",
      train_sample=args.webqa_sample,
      eval_sample=50 if args.debug else 3000, eval_setup=webqq_eval
    ))
    trainer.eval_datasets.append(TrainerDataset(
      webqa_val, "webqa-val", eval_sample=1314, eval_setup=webqq_eval))
    trainer.best_model_key.append(ResultKey("accuracy", dataset_name="webqa-val"))
  if args.lesson != None:
    lesson_datasets = {'img_contrast':TrainerDataset(ImageContrastDataset('train'),"img-contrast"), "text_contrast":TrainerDataset(TextContrastDataset("train"),"text-contrast"),"mil":TrainerDataset(MILDataset('train'),'mil'),
    'synonym':TrainerDataset(SynonymDataset("train"),"synonym-train"),"vqa_cls_obj":TrainerDataset(VQA_CLS_OBJ_Dataset("train"),"vqa_cls_obj"),"vqa_act_w_obj":TrainerDataset(VQA_ACT_W_OBJ_Dataset("all"),'vqa_act_w_obj'),
    "vqa_adj_w_obj":TrainerDataset(VQA_ADJ_W_OBJ_Dataset("all"),"vqa_adj_w_obj"),"vqa_act_no_obj":TrainerDataset(VQA_ACT_NO_OBJ_Dataset("train"),"vqa_act_no_obj")}

    training_lessons = []
    lesson_dict = {'img_contrast':args.image_contrast,'text_contrast':args.text_contrast,'mil':args.mil,'synonym':args.synonym,"vqa_cls_obj":args.vqa_cls_obj,"vqa_act_w_obj":args.vqa_act_w_obj,"vqa_adj_w_obj":args.vqa_adj_w_obj,
    "vqa_act_no_obj":args.vqa_act_no_obj}
    
    for lesson in lesson_dict:
      if lesson_dict[lesson] != None or args.lesson == 'all':
        training_lessons.append(lesson_datasets[lesson])
    if len(training_lessons) >1:
      for i,lesson_dataset in enumerate(training_lessons):
        if i == 0:
          trainer.upper_bound_no_change = 2 
          trainer.num_no_change_val = 0
          logging.info(f'Running lesson {i} out of {len(training_lessons)}')
          trainer.train_datasets.append(lesson_dataset)
          vqa_setup = EvaluationSetup(
      evaluator.VqaEvaluator(),
      dict(beam_search_spec=BeamSearchSpec(1, 10))
    )
          val_samples = io.load_json_object('/data/michal5/gpv/learning_phase_data/vqa/unseen_10/val.json')
          trainer.eval_samples.append(TrainerDataset(GpvDataset(Task.VQA,"val",True),"vqa-val",eval_sample=len(val_samples),eval_setup=vqa_setup))
          #trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=len(val_samples),eval_setup=loc_setup))
          #trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
          trainer.best_model_key.append(evaluator.ResultKey("score", dataset_name="vqa-val"))
          trainer.stratify = True
          trainer.eval_loader = deepcopy(trainer.train_loader)
          trainer.train_loader.persist_workers = False
          trainer.eval_loader.persist_workers = False
          trainer.find_unused_parameters = args.find_unused

          run_trainer_from_args(trainer, model, args)
      else:
        trainer.upper_bound_no_change = 2 
        trainer.num_no_change_val = 0
        logging.info(f'Running lesson {i} out of {len(training_lessons)}')

        trainer.train_datasets = []
        trainer.train_datasets.append(lesson_dataset)
        vqa_setup = EvaluationSetup(
        evaluator.VqaEvaluator(),
          dict(beam_search_spec=BeamSearchSpec(1, 10)))
            
        val_samples = io.load_json_object('/data/michal5/gpv/learning_phase_data/vqa/unseen_10/val.json')
        trainer.eval_samples.append(TrainerDataset(GpvDataset(Task.VQA,"val",True),"vqa-val",eval_sample=len(val_samples),eval_setup=vqa_setup))
        #trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=len(val_samples),eval_setup=loc_setup))
        #trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
        trainer.best_model_key.append(evaluator.ResultKey("score", dataset_name="vqa-val"))

        trainer.train_another_model(args.output_dir)
    else:
      trainer.upper_bound_no_change = 2 
      trainer.num_no_change_val = 0
      logging.info(f'Running single training lesson')
      trainer.train_datasets.append(training_lessons[0])
      vqa_setup = EvaluationSetup(
        evaluator.VqaEvaluator(),
          dict(beam_search_spec=BeamSearchSpec(1, 10)))
            
      val_samples = io.load_json_object('/data/michal5/gpv/learning_phase_data/vqa/unseen_10/val.json')
      trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.VQA,"val",True),"vqa-val",eval_sample=len(val_samples),eval_setup=vqa_setup))
      #trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=len(val_samples),eval_setup=loc_setup))
      #trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
      trainer.best_model_key.append(evaluator.ResultKey("score", dataset_name="vqa-val"))

      trainer.stratify = True
      trainer.eval_loader = deepcopy(trainer.train_loader)
      trainer.train_loader.persist_workers = False
      trainer.eval_loader.persist_workers = False
      trainer.find_unused_parameters = args.find_unused

      run_trainer_from_args(trainer, model, args)



        
  #   trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=4857,eval_setup=loc_setup))
  #   trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))

  #   if args.lesson == 'all' or args.image_contrast != None:
  #     trainer.train_datasets.append(TrainerDataset(ImageContrastDataset('train'),"img-contrast"))
  #   if args.lesson == 'all' or args.text_contrast != None:
  #     trainer.train_datasets.append(TrainerDataset(TextContrastDataset('train'),'text-contrast'))
  #   if args.lesson == 'all' or args.mil != None:
  #     trainer.train_datasets.append(TrainerDataset(MILDataset('train'),'mil'))
  #   loc_setup = EvaluationSetup(
  #     evaluator.LocalizationEvaluator(),
  #     dict(beam_search_spec=None)
  #   )
  #   trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=4857,eval_setup=loc_setup))
  #   trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
  # # if args.image_contrast != None or args.mil != None:
  # #   loc_setup = EvaluationSetup(
  # #     evaluator.LocalizationEvaluator(),
  # #     dict(beam_search_spec=None)
  # #   )
  # #   if args.image_contrast != None:
  # #     trainer.train_datasets.append(TrainerDataset(ImageContrastDataset('train'),"img-contrast"))
  # #   if args.mil != None:
  # #      trainer.train_datasets.append(TrainerDataset(MILDataset('train'),"mil"))
  # #   trainer.eval_datasets.append(TrainerDataset(GpvDataset(Task.DETECTION, "val", True),   "det-val",eval_sample=4857,eval_setup=loc_setup))
  # #   trainer.best_model_key.append(ResultKey("AP", dataset_name="det-val"))
  # trainer.stratify = True
  # trainer.eval_loader = deepcopy(trainer.train_loader)

  # if not args.sce:
  #   for ds in trainer.train_datasets:
  #     ds.dataset.gpv_split = False
  #   for ds in trainer.eval_datasets:
  #     ds.dataset.gpv_split = False

  # trainer.train_loader.persist_workers = False
  # trainer.eval_loader.persist_workers = False
  # trainer.find_unused_parameters = args.find_unused

  # run_trainer_from_args(trainer, model, args)


if __name__ == '__main__':
  main()