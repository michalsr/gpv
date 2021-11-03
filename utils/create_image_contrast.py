import utils.io as io
import json 
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']


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
# def get_concept_image():

# def find_non_concept_images(coco_concept,num_non_concept_images):
#     #TODO

# def get_random_position(num_images):
#     #TODO

def make_coco_cat_to_web_cat():
    coco_dict = {}
    for coco_class in UNSEEN_COMBINED:
        coco_dict[coco_class] = set()
    web_train_info = io.load_json_object('/data/michal5/web_training_info/train_image_info.json')
    for entry in web_train_info:
        coco_categories = entry['coco_categories']
        if len(entry['coco_categories']['seen']) != 0:
            for c in entry['coco_categories']['seen']:
                if c in UNSEEN_COMBINED:
                    coco_dict[c].add(entry['bing_query'])
        if len(entry['coco_categories']['unseen']) != 0:
            for c in entry['coco_categories']['unseen']:
                if c in UNSEEN_COMBINED:
                    coco_dict[c].add(entry['bing_query'])
    io.dump_json_object('/data/michal5/web_training_info/'+'coco_cat_to_web_cat.json')
# def main():
#     #TODO
if __name__ == '__main__':
    make_coco_cat_to_web_cat()