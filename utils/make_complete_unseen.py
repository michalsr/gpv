import json 
import utils.io as io
import os

UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass']
UNSEEN2 = ['banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']

def save_entry(prefix,dictionary,file_name):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    io.dump_json_object(dictionary,prefix+'/'+file_name+'.json')
def make_20():
    dist = []
    ids = set()
    prefix = '/data/michal5/gpv/learning_phase_data/web_20_temp'
    file = io.load_json_object(prefix+'/test.json')
    for f in file:
        cat = []
        if len(f['coco_categories']['seen']) != 0:
            for c in f['coco_categories']['seen']:
                cat.append(c)
        if len(f['coco_categories']) != 0:
            for c in f['coco_categories']['unseen']:
                cat.append(c)
        if len(cat) != 0:
            if  all(c in UNSEEN_COMBINED for c in cat) and int(f["id"]) not in ids:
                dist.append(f)
                ids.add(int(f["id"]))
    io.dump_json_object(dist,'/data/michal5/gpv/learning_phase_data/web_20/test.json')
def main():
    category_id = []
    unseen_categories= {}
    category_to_image_id = {}
    image_ids = {'train':{'image_ids':[]},'val':{'image_ids':[]},'test':{'image_ids':[]}}
    list_unseen = io.load_json_object('/data/michal5/gpv/learning_phase_data/split_coco_categories/category_split.json')
    list_of_held_out = ['held_from_vqa','held_from_det']
    for k in list_unseen:
        if k in list_of_held_out:
            categories = list_unseen[k]
            for c in categories:
                category_id.append(c['id'])
                unseen_categories[c['id']] = c['name']
    print(unseen_categories)
    # save_entry('/data/michal5/gpv/learning_phase_data/split_coco_categories',unseen_categories,'held_out_all')

    # gpv_classification = io.load_json_object(file_name)
    # unseen_classification = []
    
    # for entry in gpv_classification:


    #     if int(entry['category_id']) not in category_id:
            
    #         unseen_classification.append(entry)
    # print(len(unseen_classification))

    sub_folders = ['coco_classification','coco_detection','vqa','coco_captions']
    #sub_folders = ['coco_classification']
    training_type = ['test']

    for folder in sub_folders:
     
        for data_type in training_type:
            new_split = []
            ids_used = set()
            gpv_split = io.load_json_object('/data/michal5/gpv/learning_phase_data/'+folder+'/gpv_split/'+data_type+'.json')
         
            for entry in gpv_split:
                if 'category_id' in entry:
                    if int(entry['category_id']) in unseen_categories.keys() and int(entry['id']) not in ids_used:
                        new_split.append(entry)
                        ids_used.add(int(entry['id']))

                elif 'coco_categories' in entry:
                    if folder =='coco_captions':
                        entry_id = entry['cap_id']
                    elif folder == 'vqa':
                        entry_id = entry['question_id']
                    in_unseen = False
                    coco_classes = []
                    if len(entry['coco_categories']['seen']) != 0:
                        for c in entry['coco_categories']['seen']:
                            coco_classes.append(c)
                    if len(entry['coco_categories']) != 0:
                        for c in entry['coco_categories']['unseen']:
                            coco_classes.append(c)
                    if len(coco_classes) != 0:
                        if  all(c in unseen_categories.values() for c in coco_classes) and int(entry_id) not in ids_used:
                            new_split.append(entry)
                            ids_used.add(int(entry_id))


                if folder == 'coco_classification':
                    if entry['image']['image_id'] not in image_ids[data_type]['image_ids']:
                        image_ids[data_type]['image_ids'].append(entry['image']['image_id'])
            save_entry('/data/michal5/gpv/learning_phase_data/'+folder+'/held_out_test',new_split,data_type)
        # if folder == 'coco_classification':
        #     save_entry('/data/michal5/gpv/learning_phase_data/split_coco_images/held_out_all',image_ids['train'],'train')
        #     save_entry('/data/michal5/gpv/learning_phase_data/split_coco_images/held_out_all',image_ids['val'],'val')
        #     save_entry('/data/michal5/gpv/learning_phase_data/split_coco_images/held_out_all',image_ids['test'],'test')
            

if __name__ == '__main__':
  make_20()