import json 
import utils.io as io
import os


def save_entry(prefix,dictionary,file_name):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    io.dump_json_object(dictionary,prefix+'/'+file_name+'.json')

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
    save_entry('/data/michal5/gpv/learning_phase_data/split_coco_categories',unseen_categories,'held_out_all')

    # gpv_classification = io.load_json_object(file_name)
    # unseen_classification = []
    
    # for entry in gpv_classification:


    #     if int(entry['category_id']) not in category_id:
            
    #         unseen_classification.append(entry)
    # print(len(unseen_classification))

    #sub_folders = ['coco_classification','coco_detection','vqa','coco_captions']
    sub_folders = ['coco_classification']
    training_type = ['train','val','test']

    for folder in sub_folders:
     
        for data_type in training_type:
            new_split = []
            gpv_split = io.load_json_object('/data/michal5/gpv/learning_phase_data/'+folder+'/gpv_split/'+data_type+'.json')
         
            for entry in gpv_split:
                if 'category_id' in entry:
                    if int(entry['category_id']) not in unseen_categories.keys():
                        new_split.append(entry)
                if 'coco_categories' in entry:
                    in_unseen = False
                    coco_classes = []
                    if len(entry['coco_categories']['seen']) != 0:
                        for c in entry['coco_categories']['seen']:
                            coco_classes.append(c)
                    if len(entry['coco_categories']) != 0:
                        for c in entry['coco_categories']['unseen']:
                            coco_classes.append(c)
                    if not all(c in unseen_categories.values() for c in coco_classes):
                        new_split.append(entry)


                if folder == 'coco_classification':
                    if entry['image']['image_id'] not in image_ids[data_type]['image_ids']:
                        image_ids[data_type]['image_ids'].append(entry['image']['image_id'])
            save_entry('/data/michal5/gpv/learning_phase_data/'+folder+'/held_out_all',new_split,data_type)
        if folder == 'coco_classification':
            save_entry('/data/michal5/gpv/learning_phase_data/split_coco_images/held_out_all',image_ids['train'],'train')
            save_entry('/data/michal5/gpv/learning_phase_data/split_coco_images/held_out_all',image_ids['val'],'val')
            save_entry('/data/michal5/gpv/learning_phase_data/split_coco_images/held_out_all',image_ids['test'],'test')
            

if __name__ == '__main__':
  main()