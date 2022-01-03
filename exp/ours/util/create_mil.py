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
print(coco_cat_to_web.keys())
def get_concept_image(concept,num_images):
    options = web_cat_to_image_id[concept]
    chosen_images = np.random.choice(options,num_images)
    return chosen_images.tolist()

def get_another_concept(given_concept):
    new_concept_list = UNSEEN_COMBINED.copy()
    new_concept_list.remove(given_concept)
    return np.random.choice(new_concept_list,1)

def find_non_concept_images(coco_concept,num_non_concept_images):
    other_concept_image_ids = []
    found_enough_concepts = False 
    while not found_enough_concepts:
        other_concepts = np.random.choice(all_web_concepts,num_non_concept_images-1)
        is_coco_concept_in_other_concepts = []
        for c in other_concepts.tolist():
            if c in coco_cat_to_web[coco_concept]:
                is_coco_concept_in_other_concepts.append(True)
            else:
                is_coco_concept_in_other_concepts.append(False)
        if all(is_coco_concept_in_other_concepts)!= True:
            found_enough_concepts=True
    print(f'Other concepts:{other_concepts}')
    for c in other_concepts.tolist():
        image_id = get_concept_image(c,1)
        other_concept_image_ids.append(image_id[0])
    return other_concept_image_ids
def get_random_position(num_images):
    return np.random.choice(num_images,1)

def make_coco_cat_to_web_cat():
    coco_dict = {}
    final_coco_dict = {}
    for coco_class in UNSEEN_COMBINED:
    #or coco_class in UNSEEN_COMBINED:
        coco_dict[coco_class] = set()
    web_train_info = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/train_image_info.json')
    for entry in web_train_info:
        coco_categories = entry['coco_categories']
        if len(entry['coco_categories']['seen']) != 0:
            for c in entry['coco_categories']['seen']:
                if c in UNSEEN_COMBINED:
                #if c in UNSEEN_COMBINED:
                    coco_dict[c].add(entry['bing_query'])
        if len(entry['coco_categories']['unseen']) != 0:
            for c in entry['coco_categories']['unseen']:
                if c in UNSEEN_COMBINED:
                #if c in UNSEEN_COMBINED:
                    coco_dict[c].add(entry['bing_query'])
    for f in coco_dict:
        final_coco_dict[f] = list(coco_dict[f])
    io.dump_json_object(final_coco_dict,'/shared/rsaas/michal5/gpv_michal/web_training_info/'+'coco_cat_to_web_cat_seen_2.json')
def main():
    mil_data = []
    id_begin = 0
    #for i,coco_class in enumerate(SEEN):
    ids_used = set()
    contrast_group = 0

    for i,coco_class in enumerate(UNSEEN_COMBINED):
        if coco_class != 'remote':
      
            for web_c in tqdm(coco_cat_to_web[coco_class]):
            #web_class_to_search = np.random.choice(coco_cat_to_web[coco_class],1)
                for web_img in tqdm(web_cat_to_image_id[web_c]):
                    for correct in [0,1]:
                        entry = {}
                        #print(correct,'correct')
                        if correct == 0:
                            print('hi')
                            new_concept = get_another_concept(coco_class).tolist()[0]
                            print(new_concept,'new concep')
                            entry['rel_query'] = new_concept 
                            entry['query'] = f'Localize the {new_concept}'
                        else:
                            print('second hi')
                            entry['rel_query'] = coco_class 
                            entry['query'] = f'Localize the {coco_class}'
                       
                        entry['image'] = {'image_id':web_img}
                        #entry['query'] = f'Localize the {coco_class}'
                        entry['boxes'] = [0.0,0.0,1.0,1.0]
                        #entry['rel_query'] = coco_class 
                        entry['answer'] = correct 
                        entry['correct'] = correct 
                        entry['gpv_id'] = f"mil-{str(coco_class)}-{str(web_c)}-{str(web_img)}-{str(id_begin)}"
                        id_begin += 1
                        mil_data.append(entry)
    io.dump_json_object(mil_data,'/data/michal5/gpv/mil/train_large.json')
    io.dump_json_object(mil_data,'/shared/rsaas/michal5/gpv_michal/mil/train_large.json')
    




if __name__ == '__main__':
    # make_coco_cat_to_web_cat()
    # coco_cat_to_web = io.load_json_object('/data/michal5/web_training_info/coco_cat_to_web_cat_seen_2.json')
    # print(coco_cat_to_web,'coco cat to web')
    main()