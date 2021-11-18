import utils.io as io
import json 
import numpy as np
from tqdm import tqdm 
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
SEEN = ['dog']

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
GROUP_SIZE = 16


web_cat_to_image_id = io.load_json_object('/data/michal5/web_training_info/category_to_image_id.json')
all_web_concepts = list(web_cat_to_image_id.keys())
coco_cat_to_web = io.load_json_object('/data/michal5/web_training_info/coco_cat_to_web_cat.json')
def get_concept_image(concept,num_images):
    options = web_cat_to_image_id[concept]
    chosen_images = np.random.choice(options,num_images)
    return chosen_images.tolist()

def pick_web_concept_image(images):
    return np.random.choice(images,1)

def find_non_concept_images(web_concept,num_non_concept_images):
    other_concept_image_ids = []
    new_web_concept_list = all_web_concepts.copy()
    new_web_concept_list.remove(web_concept)
    other_concepts = np.random.choice(new_web_concept_list,num_non_concept_images-1)

    print(f'Other concepts:{other_concepts}')
    for c in other_concepts.tolist():
        image_id = get_concept_image(c,1)
        other_concept_image_ids.append(image_id[0])
    return other_concept_image_ids
def get_random_position(num_images):
    return np.random.choice(num_images,1)


def main():
    image_contrast_data = []
    #for i,coco_class in enumerate(SEEN):
    ids_used = set()
    contrast_group = 0
    for i,web_class in enumerate(all_web_concepts):
        #randomly choose web image 
        web_image = pick_web_concept_image(web_cat_to_image_id[web_class]).tolist()[0]
        other_images = find_non_concept_images(web_class,GROUP_SIZE)
        #randomly choose other web classes to get+ corresponding images 
  
        web_image_position = get_random_position(GROUP_SIZE)
        other_images.insert(web_image_position.tolist()[0],web_image)
        for j,img in enumerate(other_images):
            entry = {}
            entry['image'] = {'image_id':img}
            entry['query'] = f'Localize the {web_class}'
            entry['boxes'] = [0.0,0.0,1.0,1.0]
            entry['contrast_group'] = contrast_group
            entry['answer'] = str(web_image_position.tolist()[0])
            entry['gpv_id'] = f"img-contrast-{str(web_class)}-{str(contrast_group)}-{str(web_image)}-{str(img)}"
            if entry['gpv_id'] in ids_used:
                break
            ids_used.add(entry['gpv_id'])
            if j != web_image_position.tolist()[0]:
                entry['is_in_category'] = False 
            else:
                entry['is_in_category'] = True
            image_contrast_data.append(entry)
        contrast_group += 1
    io.dump_json_object(image_contrast_data,'/data/michal5/image_contrast/train_web_large.json')
    




if __name__ == '__main__':
    # make_coco_cat_to_web_cat()
    # coco_cat_to_web = io.load_json_object('/data/michal5/web_training_info/coco_cat_to_web_cat_seen.json')

    main()