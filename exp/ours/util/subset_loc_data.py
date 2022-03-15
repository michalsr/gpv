import json 
import os 
import numpy as np
import random
import utils.io as io 
UNSEEN_1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
'sheep', 'suitcase', 'surfboard', 'wine glass']
UNSEEN_2 = ['banana','baseball bat','bottle','broccoli','donut','hot dog','keyboard','laptop','train','tv']
coco_categories = io.load_json_object('/home/michal/gpv_michal/exp/ours/data/coco_categories.json')
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
SEEN = []

COCO_ID_TO_CATEGORY = {x["id"]: x["name"] for x in coco_categories}
for x in coco_categories:
    if x['name'] not in UNSEEN_1:
        if x['name'] not in UNSEEN_2:
            SEEN.append(x['name'])
COCO_CATEGORIES = list(COCO_ID_TO_CATEGORY.values())
COCO_CATEGORIES_TO_ID = {k: i for i, k in enumerate(COCO_CATEGORIES)}
COCO_CAT_TO_SUPER_CAT = {}
for k in coco_categories:
    COCO_CAT_TO_SUPER_CAT[k["name"]] = k["supercategory"]

def create_loc_subset():
    web_cat_to_image_id = io.load_json_object('/data/michal5/web_training_info/category_to_image_id.json')
    all_web_concepts = list(web_cat_to_image_id.keys())
    coco_cat_to_web = io.load_json_object('/data/michal5/web_training_info/coco_cat_to_web_cat.json')
    small_data = []
    for c in UNSEEN_COMBINED:
        relevant_web_cat = coco_cat_to_web[c]
        web_cat = np.random.choice(relevant_web_cat,1).tolist()[0]
        images = web_cat_to_image_id[web_cat]
        chosen_image = np.random.choice(images,1).tolist()[0]
        small_data.append((c,chosen_image))
        io.dump_json_object(small_data,'/home/michal/gpv_michal/lessons/small_num_localization_lessons.json')
def convert_to_gpv_format():
    updated_entries = []
    data = io.load_json_object('/home/michal/gpv_michal/lessons/small_num_localization_lessons.json')
    print(len(data))
    for i in range(len(data)):
        
        entry_1 = data[i]

        image_id = entry_1[1]
        coco_cat = entry_1[0]
        super_cat = COCO_CAT_TO_SUPER_CAT[coco_cat]
      
        
        new_entry_1 = {'boxes':[[0.0,0.0,1.0,1.0]],'category_id':COCO_CATEGORIES_TO_ID[coco_cat],"category_name":coco_cat,"coco_categories":{"seen":[],"unseen":[coco_cat]},
        "id":i,"image":{"image_id":image_id},"query":f"Localize the {coco_cat}"}
        new_entry_2 = {'boxes':[[0.0,0.0,1.0,1.0]],'category_id':COCO_CATEGORIES_TO_ID[coco_cat],"category_name":coco_cat,"coco_categories":{"seen":[],"unseen":[coco_cat]},
        "id":i,"image":{"image_id":image_id},"query":f"Localize the {super_cat}"}
        updated_entries.append(new_entry_1)
        updated_entries.append(new_entry_2)
    io.dump_json_object(updated_entries,'/home/michal/gpv_michal/coco_det_val_all/vis/val.json')
def create_loc_coco_subset():
    file_names = ['unseen_group_1','unseen_group_2','seen']
    file_to_dict_list = {'unseen_group_1':UNSEEN_1,'unseen_group_2':UNSEEN_2,'seen':SEEN}
    
    unseen_1_entries = []
    unseen_2_entries = []
    seen_entries = []
    file_to_entries = {'unseen_group_1':unseen_1_entries,'unseen_group_2':unseen_2_entries,'seen':seen_entries}
    for f in file_names:
        large_list = io.load_json_object(f'/home/michal/gpv_michal/coco_det_val_all/{f}_single_phrase/val.json')
        random.shuffle(large_list)
        for entry in large_list:
            if all(file_to_dict_list[f]) in [e['category_name'] for e in file_to_entries[f]]:
                break 
            if entry['category_name'] not in [e['category_name'] for e in file_to_entries[f]]:
                file_to_entries[f].append(entry)
    io.dump_json_object(unseen_1_entries,'/home/michal/gpv_michal/coco_det_val_all/unseen_group_1_small/val.json')
    io.dump_json_object(unseen_2_entries,'/home/michal/gpv_michal/coco_det_val_all/unseen_group_2_small/val.json')
    io.dump_json_object(seen_entries,'/home/michal/gpv_michal/coco_det_val_all/seen_small/val.json')
def cat_to_coco_entry(list_of_entries):
    entries = {}
    for entry in list_of_entries:
        if entry['category_name'] not in entries:
            entries[entry['category_name']] = []
        entries[entry['category_name']].append(entry)
    return entries
def fix_coco_stuff():
    new_entries = []
    coco_cat_to_entries = cat_to_coco_entry(io.load_json_object('/home/michal/gpv_michal/coco_det_val_all/unseen_group_2_single_phrase/val.json'))
    unseen_group_2_small = io.load_json_object('/home/michal/gpv_michal/coco_det_val_all/unseen_group_2_small/val.json')
    c_to_fix = ['bottle']
    print(coco_cat_to_entries.keys())
    for entry in unseen_group_2_small:
        if entry['category_name'] not in c_to_fix:
            new_entries.append(entry)
        else:
            new_to_add = np.random.choice(coco_cat_to_entries[entry['category_name']])
            new_entries.append(new_to_add)
    io.dump_json_object(new_entries,'/home/michal/gpv_michal/coco_det_val_all/unseen_group_2_small/val.json')

if __name__ == '__main__':
   fix_coco_stuff()
   


