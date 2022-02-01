import json 
import utils.io as io 
from tqdm import tqdm 
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
adj_categories = io.load_json_object('/shared/rsaas/michal5/gpv_michal/exp/ours/data/webqa_adj_types.json')

def create_unseen_cat_to_verb_and_adj_type(web_training):
    #print(adj_categories.keys())
    adj_cats = {}
    verb_cats= {}
    for c in UNSEEN_COMBINED:
        adj_cats[c] = {}
        verb_cats[c] = {}
    for entry in web_training:
        coco_classes = entry['coco_categories']
        all_coco_classes = []
        if len(coco_classes['seen'])!=0:
            for c in coco_classes['seen']:
                if c in UNSEEN_COMBINED:
                    all_coco_classes.append(c)
        if len(coco_classes['unseen']) != 0:
            for c in coco_classes['unseen']:
                if c in UNSEEN_COMBINED:
                    all_coco_classes.append(c)
        if len(all_coco_classes) != 0:
            for c in all_coco_classes:
                class_adj = adj_cats[c]
                class_verb = verb_cats[c]
                print(entry)
                if entry['verb'] != 'null' and entry['verb'] != None:
                    if entry['verb'] not in class_verb:
                        class_verb[entry['verb']] = []
                    class_verb[entry['verb']].append(entry['image']['image_id'])
                if entry['adj'] != 'null' and entry['adj'] != None:
                    print(type(adj_categories[entry['adj']]))
                    print(adj_categories[entry['adj']],entry['adj'])
                    print(adj_categories.keys())
                    if adj_categories[entry['adj']] not in class_adj:
                        class_adj[entry['adj']] = []
                    class_adj[entry['adj']].append(entry['image']['image_id'])
    io.dump_json_object(adj_cats,'/shared/rsaas/michal5/gpv_michal/web_training_info/adj_category_classification.json')
    io.dump_json_object(verb_cats,'/shared/rsaas/michal5/gpv_michal/web_training_info/verb_category_classification.json')
def create_categories_dict(web_training):
    category_dict = {}
    for entry in web_training:
        query = entry['bing_query']
        if query not in category_dict:
            category_dict[query] = []
    return category_dict
def all_coco_to_web(web_training):
    coco_to_web = {}
    num_entries = 0
    for entry in tqdm(web_training):
        print(num_entries,'num entries')
        coco_classes_in_img = set()
        coco_classes = entry['coco_categories']
        if len(coco_classes['seen']) != 0:
            for c in coco_classes['seen']:
                if c not in coco_to_web:
                    coco_to_web[c] = []
                coco_to_web[c].append(entry['bing_query'])
                num_entries+= 1
        if len(coco_classes['unseen']) != 0:
            for c in coco_classes['unseen']:
                if c not in coco_to_web:
                    coco_to_web[c] = []
                coco_to_web[c].append(entry['bing_query'])
                num_entries += 1

    #print(len(coco_to_web.),'coco to web')
    io.dump_json_object(coco_to_web,'/shared/rsaas/michal5/gpv_michal/web_training_info/all_coco_to_web_bing_query.json')
def load_info(web_training,category_image_id,category_pos,category_coco_id):
    for entry in web_training:

        query = entry['bing_query']
        pos_dict = {'noun':entry['noun'],'verb':entry['verb'],'adj':entry['adj']}
        category_image_id[query].append(entry['image']['image_id'])
        category_pos[query] = pos_dict
        category_coco_id[query].append(entry['coco_categories'])
    io.dump_json_object(category_image_id,'exp/ours/web_training_info/category_to_image_id.json')
    io.dump_json_object(category_pos,'exp/ours/web_training_info/category_to_pos.json')
    io.dump_json_object(category_coco_id,'exp/ours/web_training_info/category_coco_id.json')
    


if __name__ == '__main__':
    web_training = io.load_json_object('web_training_info/train_image_info.json')
    create_unseen_cat_to_verb_and_adj_type(web_training)
