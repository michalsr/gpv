import utils.io as io
import json 
import os 
import numpy as np
from tqdm import tqdm 
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
# SEEN = ['dog']

#go through each concept 
#pick image related to concept 
#find group of images not in concept
#assign number to group
#randomly choose position for target image
#each entry has 
#     all information in web entry
#     group number
#     whether it belongs to the class
#     query of locate class
#     answer is str of randomly chosen position for target image 
IMAGES_PER_COCO_CONCEPT = 1
GROUP_SIZE = 11


web_cat_to_image_id = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/category_to_image_id.json')
all_web_concepts = list(web_cat_to_image_id.keys())
coco_cat_to_web = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/coco_cat_to_web_cat.json')
def get_concept_image(concept,num_images):
    options = web_cat_to_image_id[concept]
    chosen_images = np.random.choice(options,num_images)
    return chosen_images.tolist()


def get_other_unseen_classes(coco_concept,num_classes):
    unseen_removed_current_concept = UNSEEN_COMBINED.copy()
    unseen_removed_current_concept.remove(coco_concept)
    return np.random.choice(unseen_removed_current_concept,num_classes,replace=False).tolist()
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
    # for c in other_concepts.tolist():
    #     image_id = get_concept_image(c,1)
    #     other_concept_image_ids.append(image_id[0])
    return other_concepts.tolist()
def get_random_position(num_images):
    return np.random.choice(num_images,1)


def main():
    text_contrast_data = []
    #for i,coco_class in enumerate(SEEN):
    ids_used = set()
    contrast_group = 0
    id_num = 0
    for i,coco_class in enumerate(UNSEEN_COMBINED):
        if coco_class != 'remote':
            for web_c in tqdm(coco_cat_to_web[coco_class]):
            #web_class_to_search = np.random.choice(coco_cat_to_web[coco_class],1)
                for web_img in tqdm(web_cat_to_image_id[web_c]):
                    # if web_img in os.listdir('/data/michal5/gpv/learning_phase_data/web_data/images/'):
                    #web_class_images = get_concept_image(web_class_to_search.tolist()[0],IMAGES_PER_COCO_CONCEPT)
                    #print(f'Web class images:{web_class_images}')
                    #other_concepts = find_non_concept_images(coco_class,GROUP_SIZE)
                    other_concepts = get_other_unseen_classes(coco_class,GROUP_SIZE-1)
                    #print(f'Other image ids:{other_images}')
                    web_concept_position = get_random_position(GROUP_SIZE)
                    other_concepts.insert(web_concept_position.tolist()[0],coco_class)
                    for j,concept_text in enumerate(other_concepts):
                        entry = {}
                        img = get_concept_image(concept_text,1)
                        entry['image'] = {'image_id':web_img}
                        #entry['image'] = {'image_id':img[0]}
                        entry['query'] = f'Localize the {concept_text}'
                        entry['boxes'] = [0.0,0.0,1.0,1.0]
                        entry['contrast_group'] = contrast_group
                        entry['rel_query'] = coco_class
                        entry['answer'] = str(web_concept_position.tolist()[0])
                        entry['gpv_id'] = f"text-contrast-{str(coco_class)}-{str(web_c)}-{str(web_img)}-{str(id_num)}"
                        id_num += 1
                        if entry['gpv_id'] in ids_used:
                            break
                        ids_used.add(entry['gpv_id'])
                        if j != web_concept_position.tolist()[0]:
                            entry['is_in_category'] = False 
                        else:
                            entry['is_in_category'] = True
                        text_contrast_data.append(entry)
                    contrast_group += 1
    print(len(text_contrast_data),'text contrast data')
    io.dump_json_object(text_contrast_data,'/data/michal5/gpv/lessons/text_contrast_train_15_per_group.json')
    io.dump_json_object(text_contrast_data,'/shared/rsaas/michal5/gpv_michal/lessons/text_contrast_train_15_per_group.json')

    #io.dump_json_object(text_contrast_data,'/shared/rsaas/michal5/gpv_michal/text_contrast_train_large.json')




if __name__ == '__main__':
    # make_coco_cat_to_web_cat()
    # coco_cat_to_web = io.load_json_object('/data/michal5/web_training_info/coco_cat_to_web_cat_seen.json')

    main()