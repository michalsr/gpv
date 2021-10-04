import json 
import utils.io as io
UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']


web_20_entries = []
web_80 = io.load_json_object('/home/michal/gpv_michal/exp/ours/web_training_info/test_image_info.json')

for entry in web_80:
    classes = set()
    coco_categories = entry['coco_categories']
    if len(coco_categories['seen']) != 0:
        for c in coco_categories['seen']:
            classes.add(c)
    if len(coco_categories['unseen']) != 0:
        for c in coco_categories['unseen']:
            classes.add(c)
    if all(c in UNSEEN1 for c in classes):
        web_20_entries.append(entry)
print(len(web_20_entries))
io.dump_json_object(web_20_entries,'/data/michal5/gpv/learning_phase_data/web_20/test_image_info.json')