import json 
import utils.io as io 
unseens = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
new_list = []
coco_entries = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/original_split/train.json')
for entry in coco_entries:
    if entry['category_name'] in unseens:
        new_list.append([entry['category_name'],entry['image']['image_id']])
io.dump_json_object(new_list,'/shared/rsaas/michal5/gpv_michal/lessons/coco_web_data.json')