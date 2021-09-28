import os
import shutil

file_categories = ['coco_captions','coco_classification','coco_detection','coco_detection']
data_type = ['train','val','test']
for f in file_categories:
    for d in data_type:
        location = '/shared/rsaas/michal5/gpv_michal/held_out_json/'+f+'/'+'held_out_all/'+d+'.json'
        destination = '/data/michal5/gpv/learning_phase_data/'+f+'/'+'held_out_all/'+d+'.json'
        shutil.copy(location,destination)