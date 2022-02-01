import utils.io as io
import json 
import numpy as np
from tqdm import tqdm 
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']


web_cat_to_image_id = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/category_to_image_id.json')
all_web_concepts = list(web_cat_to_image_id.keys())
coco_cat_to_web = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/coco_cat_to_web_cat.json')
def main():

    global_id = 0       
    synonym_data = []
    coco_to_super = {}
    #map coco category to super catogorey 
    coco_list = io.load_json_object('/shared/rsaas/michal5/gpv_michal/exp/ours/data/coco_categories.json')
    for c in coco_list:
        if c['name'] not in coco_to_super:
            if c['name'] in UNSEEN_COMBINED:
                coco_to_super[c['name']] = c['supercategory']
    ids_used = set()
    contrast_group = 0
    for i,coco_class in enumerate(UNSEEN_COMBINED):
        if coco_class != 'remote':
            for web_c in tqdm(coco_cat_to_web[coco_class]):
            #web_class_to_search = np.random.choice(coco_cat_to_web[coco_class],1)
                for web_img in tqdm(web_cat_to_image_id[web_c]):
                    
                    
                    entry_1 = {}
                    entry_1['image'] = {'image_id':web_img}
                    entry_1['query'] = f'Localize the {coco_class}'
                    entry_1['boxes'] = [0.0,0.0,1.0,1.0]
                    entry_1['rel_query'] = coco_class
                    
                    entry_1['gpv_id'] = f"synonym-{str(coco_class)}-{str(web_img)}-{str(global_id)}"
                    global_id += 1
                    if entry_1['gpv_id'] in ids_used:
                        break
                    ids_used.add(entry_1['gpv_id'])
                    entry_2 = {}
                    entry_2['image'] = {'image_id':web_img}
                    entry_2['query'] = f'Localize the {coco_to_super[coco_class]}'
                    entry_2['boxes'] = [0.0,0.0,1.0,1.0]
                    entry_2['rel_query'] = coco_to_super[coco_class]
                    entry_2['gpv_id'] = f"synonym-{str(coco_class)}-{str(web_img)}-{str(coco_to_super[coco_class])}-{str(global_id)}"
                    global_id += 1
                    if entry_2['gpv_id'] in ids_used:
                        break
                    ids_used.add(entry_2['gpv_id'])
                    entry_1['answer'] = entry_2['gpv_id']
                    entry_2['answer'] = entry_1['gpv_id']
                    synonym_data.append(entry_1)
                    synonym_data.append(entry_2)

                    
               
    io.dump_json_object(synonym_data,'/data/michal5/gpv/lessons/synonym_train_super_rel.json')
    io.dump_json_object(synonym_data,'/shared/rsaas/michal5/gpv_michal/lessons/synonym_train_super_rel.json')




if __name__ == '__main__':
    # make_coco_cat_to_web_cat()
    # coco_cat_to_web = io.load_json_object('/data/michal5/web_training_info/coco_cat_to_web_cat_seen.json')

    main()