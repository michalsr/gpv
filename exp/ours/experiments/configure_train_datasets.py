from typing import Type
from exp.ours.train.lesson_trainer import TrainerDataset, RunArgs, Trainer, EvaluationSetup
from exp.ours.data.image_contrast import ImageContrastDataset
from exp.ours.data.text_contrast import TextContrastDataset
from exp.ours.data.synonym import SynonymDataset 
from exp.ours.data.mil import MILDataset
import random 
import numpy as np 
import utils.io as io 
CONTRAST_GROUP = 0
ID = 0 
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']


def create_training_datasets(data,sampled_lessons,batch_size,map_int_to_lesson,lesson_datasets,combine_same_lesson=False):
    training_datasets = []
    lesson_names = {'image_contrast':convert_to_contrast,'text_contrast':convert_to_text_contrast,'synonym':convert_to_synonym,
    'mil':convert_to_mil}
    lesson_datasets = {'image_contrast':ImageContrastDataset, "text_contrast":TextContrastDataset,"mil":MILDataset,
    'synonym':SynonymDataset}

    lesson_count = {'image_contrast':0,'text_contrast':0,'mil':0,'synonym':0}
    num_split = len(data)/batch_size
    new_data = np.split(np.array(data),num_split)
    print(type(new_data))
    assert len(new_data) >= len(sampled_lessons)
    for i in sampled_lessons:
        lesson_count[map_int_to_lesson[i]] += 1
  
    
    num_start = 0 
    num_end = 0
    # if combine_same_lesson:
    #     for l in lesson_count:
    #         print(num_start,'num start')
    #         if lesson_count[l] > 0:
    #             total_data = batch_size*lesson_count[l]
    #             if num_start+ total_data > len(data):
    #                 lesson_data = data[num_start:]
    #             else:
    #                 lesson_data = data[num_start:num_start+total_data]
    #             num_start = num_start + total_data
    #             final_data = lesson_names[l](lesson_data)
    #             final_dataset = lesson_datasets[l](split='train',raw_instances=final_data)
    #             training_datasets.append(TrainerDataset(final_dataset,l))
    # else:
    random.shuffle(new_data)
    id_value = 0
    contrast_group = 0
    for i,lesson in enumerate(sampled_lessons):
        lesson_data = new_data[i]

  
        final_data,contrast_group,id_value = lesson_names[map_int_to_lesson[lesson]](lesson_data,contrast_group,id_value)
        final_dataset = lesson_datasets[map_int_to_lesson[lesson]](split='train',raw_instances=final_data)
        training_datasets.append(TrainerDataset(final_dataset,map_int_to_lesson[lesson]))
    return training_datasets



def convert_to_contrast(sampled_data,contrast_group,id_value):
    final_data = []
    correct_index = np.random.choice(len(sampled_data))
    correct_class = sampled_data[correct_index][0]
    for i,entry in enumerate(sampled_data):
        entry = entry.tolist()
        c = entry[0]
        img = entry[1]

        
        sample_entry = {}
        sample_entry['image'] = {'image_id':img}
        if i != correct_index:
            options = UNSEEN_COMBINED.copy()
            options.remove(c)
            img_text = np.random.choice(options).tolist()
     
            sample_entry['query'] = f'Localize the {img_text}'
        else:
            sample_entry['query'] = f'Localize the {c}'

        sample_entry['boxes'] = [0.0,0.0,1.0,1.0]
        sample_entry['contrast_group'] = contrast_group
        sample_entry['answer'] = str(correct_index)
        sample_entry['gpv_id'] = f"contrast-{c}-correct-{str(correct_index)}-{str(contrast_group)}-{str(id_value)}"
        sample_entry['rel_query'] = correct_class
        id_value += 1
        final_data.append(sample_entry)
    contrast_group += 1 
    return final_data, contrast_group, id_value  
    







def convert_to_image_contrast(sampled_data,batch_size):
    image_list = []
    
    final_data_list = []
    contrast_group = 0
    id_to_use = 0
    for data_entry in sampled_data:

        if len(data_entry)>1:
            for coco_class in data_entry.keys():
                list_of_images = list(data_entry.values())
                random.shuffle(list_of_images)
                correct_img = list_of_images.index(data_entry[coco_class])
                for web_img in list_of_images:
                    entry = {}
                    entry['image'] = {'image_id':web_img}
                    entry['query'] = f'Localize the {coco_class}'
             
                    entry['boxes'] = [0.0,0.0,1.0,1.0]
                    entry['contrast_group'] = contrast_group
                    entry['answer'] = str(correct_img)
                    entry['gpv_id'] = f"img-contrast-{str(coco_class)}-{str(web_img)}-{str(id_to_use)}"
                    id_to_use+= 1
                    entry['rel_query'] = coco_class
                    final_data_list.append(entry)
                contrast_group += 1
   
    if len(final_data_list) == 0:
        raise TypeError
    return final_data_list

 

def convert_to_text_contrast(sampled_data):
    final_data_list = []
    contrast_group = 0
    id_to_use = 0
    for data_entry in sampled_data:
     
        if len(data_entry)>1:
            for coco_class in data_entry.keys():
    
                list_of_classes = list(data_entry.keys())
                random.shuffle(list_of_classes)
                correct_class = list_of_classes.index(coco_class)
                for other_class in list_of_classes:
                    entry = {}
                    entry['image'] = {'image_id':data_entry[coco_class]}
                    entry['query'] = f'Localize the {other_class}'
            
                    entry['boxes'] = [0.0,0.0,1.0,1.0]
                    entry['contrast_group'] = contrast_group
                    entry['answer'] = str(correct_class)
                    entry['gpv_id'] = f"text-contrast-{str(coco_class)}-{str(other_class)}-{str(id_to_use)}"
                    id_to_use+= 1
                    entry['rel_query'] = coco_class
                    final_data_list.append(entry)
                contrast_group += 1

    if len(final_data_list) == 0:
        raise TypeError
    return final_data_list
def convert_to_synonym(sampled_data,contrast_group,id_value):
    synonym_data = []
    coco_to_super = {}
    global_id = 0
    #map coco category to super catogorey 
    coco_list = io.load_json_object('/shared/rsaas/michal5/gpv_michal/exp/ours/data/coco_categories.json')
    for c in coco_list:
        if c['name'] not in coco_to_super:
            if c['name'] in UNSEEN_COMBINED:
                coco_to_super[c['name']] = c['supercategory']
    for data_entry in sampled_data:
        data_entry = data_entry.tolist()
        if len(data_entry)>1:
            coco_class = data_entry[0]
            img = data_entry[1]
        
        
            entry_1 = {}
            entry_1['image'] = {'image_id':img}
            entry_1['query'] = f'Localize the {coco_class}'
    
            entry_1['boxes'] = [0.0,0.0,1.0,1.0]
            entry_1['rel_query'] = coco_class
            
            entry_1['gpv_id'] = f"synonym-{str(coco_class)}-{str(img)}-{str(id_value)}"
            id_value += 1
        
            entry_2 = {}
            entry_2['image'] = {'image_id':img}
            entry_2['query'] = f'Localize the {coco_to_super[coco_class]}'
            entry_2['boxes'] = [0.0,0.0,1.0,1.0]
            entry_2['rel_query'] = coco_to_super[coco_class]
            entry_2['gpv_id'] = f"synonym-{str(coco_class)}-{str(img)}-{str(coco_to_super[coco_class])}-{str(id_value)}"
            id_value += 1
    
            entry_1['answer'] = entry_2['gpv_id']
            entry_2['answer'] = entry_1['gpv_id']
            synonym_data.append(entry_1)
            synonym_data.append(entry_2)

    if len(synonym_data) == 0:
        raise TypeError 
    return synonym_data,contrast_group,id_value
        


def convert_to_mil(sampled_data,contrast_group,id_value):
    final_data = []
    #print(sampled_data)
    id_begin = 0

    
    for data_entry  in sampled_data:
  

        data_entry = data_entry.tolist()
        coco_class = data_entry[0]
        img = data_entry[1]
    
        correct = np.random.choice([0,1]).tolist()

        
        entry = {}
        if correct == 0:
            
            diff_classes = UNSEEN_COMBINED.copy()
       
            diff_classes.remove(coco_class)
            random_class = np.random.choice(diff_classes).tolist()
        
            assert random_class != coco_class
            entry['rel_query'] = random_class
            entry['query'] = f'Localize the {random_class}'
        else:
            entry['rel_query'] = coco_class
            entry['query'] = f'Localize the {coco_class}'
    
        entry['image'] = {'image_id':img}
        #entry['query'] = f'Localize the {coco_class}'
        entry['boxes'] = [0.0,0.0,1.0,1.0]
        #entry['rel_query'] = coco_class 
        entry['answer'] = correct 
        entry['correct'] = correct 
        entry['gpv_id'] = f"mil-{str(correct)}-{str(coco_class)}-{str(id_value)}"
        id_value += 1
        final_data.append(entry)

    if len(final_data) ==0:
        raise TypeError
    return final_data,contrast_group,id_value



