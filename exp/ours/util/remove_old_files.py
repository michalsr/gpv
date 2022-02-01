import os
from exp.ours.util.auto_select_utils import *
def remove_non_best_files(folder_name):
    auto_task = load_auto_task_params(folder_name)
    for e in range(auto_task.start_epoch):
        for j in range(auto_task.num_trajec ):
            output_file = folder_name+f'epoch_{e}_'+f'lesson_{j}/'+'r0/best-state.pth'
            if output_file != auto_task.best_model_path:
                print(folder_name+f'epoch_{e}_'+f'lesson_{j}/'+'r0/checkpoint.pth')
                if os.path.exists(folder_name+f'epoch_{e}_'+f'lesson_{j}/'+'r0/checkpoint.pth'):
                    os.remove(folder_name+f'epoch_{e}_'+f'lesson_{j}/'+'r0/checkpoint.pth')
                    os.remove(folder_name+f'epoch_{e}_'+f'lesson_{j}/'+'r0/best-state.pth')

if __name__ == '__main__':
  remove_non_best_files('outputs/auto_select/exp_5/')