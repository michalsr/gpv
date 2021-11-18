import json 
import utils.io as io
# UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
#              'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
#              'hot dog', 'keyboard', 'laptop', 'train', 'tv']
# UNSEEN2 = ['banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
#              'hot dog', 'keyboard', 'laptop', 'train', 'tv']

UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
'sheep', 'suitcase', 'surfboard', 'wine glass']

web_20_entries = []
web_80 = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/gpv_split/test.json')

for entry in web_80:

    classes = set()
    coco_categories = entry['coco_categories']
    if len(coco_categories['seen']) != 0:
        for c in coco_categories['seen']:
            classes.add(c)
    if len(coco_categories['unseen']) != 0:
        for c in coco_categories['unseen']:
            classes.add(c)
    if len(classes) != 0:
        if all(c in UNSEEN1 for c in classes):
            web_20_entries.append(entry)
print(len(web_20_entries))
print(len(web_80))
io.dump_json_object(web_20_entries,'/data/michal5/gpv/learning_phase_data/coco_detection/unseen_10/test.json')
# print(len(web_20_entries))
# io.dump_json_object(web_20_entries,'/data/michal5/gpv/learning_phase_data/web_20/test_image_info.json')
# new_train_data = []
# old_train_data = io.load_json_object('/data/michal5/gpv/learning_phase_data/web_20/test_image_info.json')
# prefixes = ["q", "1n", "1v", "1a", "2a", "2v"]
# for i,entry in enumerate(old_train_data):
#     new_entry = entry
#     new_entry['web_id'] = {}
#     for p in prefixes:
#         new_entry['web_id'][p] = f"{p}_{i}"
#     new_train_data.append(new_entry)
# io.dump_json_object(new_train_data,'/data/michal5/gpv/learning_phase_data/web_20/test_image_info.json')
old = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/gpv_split/test.json')
new = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_10/test.json')
print(len(old),'old')
print(len(new),'new')
