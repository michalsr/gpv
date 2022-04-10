import json 
import utils.io as io
import pandas as pd 
cat_to_id = {'bear':0,'bed':1,'bird':2,'car':3,'cat':4,'chair':5,'cow':6,'dog':7,'elephant':8,
'horse':9,'sheep':10}
id_to_cat = {1:'bed',9:'horse',10:'sheep'}
coco_id = {'bed':65,'horse':19,'sheep':20}
def convert_img_to_class():
    img_to_class = {}
    test_annotations = pd.read_csv('/home/michal5/pcl_pytorch/new_annotations.csv')
    for index, row in test_annotations.iterrows():
        image_id = row['Image Id'].split('.')[0]
        img_to_class[image_id] = row['Class']
    return img_to_class
web_json = io.load_json_object('/home/michal5/pcl_pytorch/web_test_annotations.json')
image_to_annotations = {}

gpv_annotation_fiels = ['query','boxes','instance ids','category_id','category_name','id']
all_examples = []
for entry in web_json['annotations']:
    if entry['category_id'] in id_to_cat:
        if entry['image_id'] not in image_to_annotations:
            image_to_annotations[entry['image_id']] = []
        image_to_annotations[entry['image_id']].append(entry)

for image in image_to_annotations:
    example = {}
    bboxes = image_to_annotations[image] 
    example['image'] = {'image_id':image}
    example['instance_ids'] = [image]
    example['id'] = [image]
    for bbox_entry in bboxes:
        single_bbox = bbox_entry['bbox']
        if 'category_id' not in example:
            category_id = coco_id[id_to_cat[bbox_entry['category_id']]]
            example['category_id'] = category_id 
        if 'category_name' not in example:
            category_name = id_to_cat[bbox_entry['category_id']]
            example['category_name'] = category_name 
        
        bbox_id = bbox_entry['id']
        if 'boxes' not in example:
            example['boxes'] = []
        example['boxes'].append(single_bbox)
    example['query'] = f'Localize the {category_name}'
    all_examples.append(example)
io.dump_json_object(all_examples,'/shared/rsaas/michal5/gpv_michal/web_imgs.json')









