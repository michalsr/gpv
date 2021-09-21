import os
from utils.io import io 

categories = io.load_json_object('/data/michal5/split_coco_categories/held_out_all.json')

types = ['train','test','val']

for t in types:
    file_list = io.load_json_object('/data/michal5/vqa/'+t+'.json')
    c = []
    for entry in file_list:
        seen_classes = entry["coco_categories"]["seen"]
        unseen_classes = entry["coco_categories"]["unseen"]
        for s in seen_classes:
            c.append(s)
        for s in unseen_classes:
            c.append(s)
        if any(c in categories.values()):
            print(entry)
        