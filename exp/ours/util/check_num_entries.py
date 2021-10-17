import json 
import utils.io as io


def count_gpv_split(split_name):
    tasks = ["coco_classification","coco_detection","coco_captions","vqa"]
    training_type = ["train","val","test"]
    entries= {"coco_classification":{"train":0,"val":0,"test":0},"coco_detection":{"train":0,"val":0,"test":0},"coco_captions":{"train":0,"val":0,"test":0},"vqa":{"train":0,"val":0,"test":0}}
    for ta in tasks:
        for t in training_type:
            list_of_tasks = io.load_json_object("/data/michal5/gpv/learning_phase_data/"+ta+"/"+split_name+"/"+t+".json")
            entries[ta][t] = len(list_of_tasks)
    print(entries)
if __name__ == '__main__':
    count_gpv_split("held_out_test")