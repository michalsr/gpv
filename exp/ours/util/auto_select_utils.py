import torch
from exp.ours.experiments.train_t5_lesson_select import *
import os 
from torch.utils.tensorboard import SummaryWriter
from exp.ours import params 
import torch.optim 
# train_tasks: Optional[List[str]] = None
#   gpv_model: Optional[T5GpvPerBox] = None 
#   args: Optional[str] = None 
#   policy_network: Optional[nn.Module] = None

#   policy_opt: Optional[nn.Module] = None 
#   lessons: Optional[List[str]] = None 
#   trainer: Optional[Trainer] = None 
#   num_trajec: Optional[int] = None 
#   sampled_lesson: Optional[int] = None
#   current_lesson_trajec: Optional[List] = None  
#   map_int_to_lesson: Optional[dict] =  None 
#   map_lesson_to_int: Optional[dict] = None 
#   best_model_path: Optional[str] = None 
#   trajec_to_validation_scores: Optional[dict] = None 
#   trajec_to_normalized_scores:Optional[dict] = None
#   trajec_to_output_dir: Optional[dict] = None
#   epochs: Optional[int] = None
#   log_prob: Optional[dict] = None 
#   auto_logger: Optional[logging.Logger] = None 
#   outer_log_step: Optional[int] = None
#   inner_log_step: Optional[int] = None
#   summary_writer: Optional[SummaryWriter] = None
#   start_epoch = 0
#   start_trajec = 0
# {'train_tasks':self.train_tasks,'args':self.args,'lessons':self.lessons,
#      'num_trajec':self.num_trajec,'current_lesson_trajec':self.current_lesson_trajec,'map_int_to_lesson':self.map_int_to_lesson,'map_lesson_to_int':self.map_lesson_to_int,
#      'best_model_path':self.best_model_path,'trajec_to_validation_scores':self.trajec_to_validation_scores,'trajec_to_normalized_scores':self.trajec_to_normalized_scores,
#      'trajec_to_output_dir':self.trajec_to_output_dir,'epochs':self.epochs,'outer_log_step':self.outer_log_step,'inner_log_step':self.inner_log_step,
#      'start_epoch':self.start_epoch,'start_trajec':self.start_trajec}
def load_auto_task_params(output_dir):
    a = AutoTask()
    auto_select_params = torch.load(output_dir+'/auto_task_chkpt.pt')
    for k in auto_select_params:
        setattr(a,k,auto_select_params[k])
    return a 
def load_policy_net(output_dir,num_lessons):
    w = Weights(num_lessons,12)
    policy_net_state_dict = torch.load(output_dir+'/policy_chkpt.pt')
    print(policy_net_state_dict)
    w.load_state_dict(policy_net_state_dict['weights'])
    w.optimizer = torch.optim.Adam(w.params) 
    w.optimizer.load_state_dict(policy_net_state_dict['optim'])
    w.log_prob = policy_net_state_dict['log_prob']
    return w
def resume_training(output_dir):

    a = load_auto_task_params(output_dir)
    print(a.start_epoch,'start epoch')
    policy = load_policy_net(output_dir,len(a.lessons))
    a.policy_network = policy 
    a.policy_opt = policy.optimizer
    if a.output_dir != None: 
        a.summary_writer = SummaryWriter(a.output_dir+'/tensorboard')
    else:
        a.summary_writer = SummaryWriter(output_dir+'/tensorboard')
    return a 

    
