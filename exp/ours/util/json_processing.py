import json 
import utils.io as io 


def create_categories_dict(web_training):
    category_dict = {}
    for entry in web_training:
        query = entry['bing_query']
        if query not in category_dict:
            category_dict[query] = []
    return category_dict

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
    web_training = io.load_json_object('exp/ours/web_training_info/train_image_info.json')
    category_to_image_id_empty = create_categories_dict(web_training)
    category_pos_empty = create_categories_dict(web_training)
    category_coco_id_empty = create_categories_dict(web_training)
    load_info(web_training,category_to_image_id_empty,category_pos_empty,category_coco_id_empty)
