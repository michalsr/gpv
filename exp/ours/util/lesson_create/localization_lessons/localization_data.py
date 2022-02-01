from itertools import permutations
from xml.etree.ElementPath import iterfind

import utils.io as io
import random
import numpy as np 

#subset of web that is in unseen classes = total
#put web_training_entries into groups of 10
#each group has a batch of 10 images
#each image is from a different coco class 

#how to convert from web entry to lesson data

#given batch of 10 images, assigned a lesson 

#image contrast 
#creates 10 image contrast examples (with 10 images each)

#text contrast 
#creates 10 text contrast examples (with 10 images each)

#mil 
#randomly assign each example to be true or false

#synonym
#creates 10 synonym examples (with 2 images each)

#for each trajec, sample DATA_SIZE/10 lessons which means DATA_SIZE/10 training datasets 
#each dataset has its own sampler 



#first map coco class directly to web images for iterator 
#iterate through unseen classes 
# each unseen class has an iterator 
#once 10 images in a batch resume 
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
web_cat_to_image_id = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/category_to_image_id.json')
coco_cat_to_web = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/coco_cat_to_web_cat.json')
def construct_unseen_to_web_direct(done_class):
    smallest_class = None 
    smallest_length = 10000000
    coco_class_to_image = {}
    for i,coco_class in enumerate(UNSEEN_COMBINED):
        if coco_class not in done_class:
            if coco_class not in coco_class_to_image:
                coco_class_to_image[coco_class] = []
            for web_c in coco_cat_to_web[coco_class]:
                for web_img in web_cat_to_image_id[web_c]:
                    coco_class_to_image[coco_class].append(web_img)
    for i in coco_class_to_image:
      
        if len(coco_class_to_image[i]) <= smallest_length:
            smallest_class = i 
            smallest_length = len(coco_class_to_image[i])
    return coco_class_to_image,smallest_class 
def construct_coco_class_iterators(coco_class_to_image):
    coco_class_iterators = {}
    for c in UNSEEN_COMBINED:
        coco_class_iterators[c] = iter(coco_class_to_image[c])
    return coco_class_iterators
def get_next_image(coco_class_to_image,coco_class_iterators,coco_class):
    try:
        img = next(coco_class_iterators[coco_class])
    except StopIteration:
        random.shuffle(coco_class_to_image[coco_class])
        coco_class_iterators[coco_class] = iter(coco_class_to_image[coco_class])
        img = next(coco_class_iterators[coco_class])
    return img, coco_class_iterators
def create_img_dict(coco_class_to_image):
    img_track = {}
    #print(coco_class_to_image['bed'])
    for k in coco_class_to_image:
        for i in coco_class_to_image[k]:
            #print(i)
            img_track[i] = 'False'
    return img_track
def check(img_tracker):
    num_false = 0
    for i in img_tracker.values():
        if i == 'False':
            num_false += 1
    return num_false
def choose_coco_class(current_list_of_classes,empty_classes):
    unseen_combined_copy = UNSEEN_COMBINED.copy()
    for c in current_list_of_classes:
        unseen_combined_copy.remove(c)
    for c in empty_classes:
        if c in unseen_combined_copy:
            unseen_combined_copy.remove(c)
    if len(unseen_combined_copy)>1:
        return np.random.choice(unseen_combined_copy)
    else:
        return None 
def choose_image(coco_class,web_direct):
    chosen_img = np.random.choice(web_direct[coco_class])
    before = len(web_direct[coco_class])
    done = False 
    if before>= 1:


        web_direct[coco_class].remove(chosen_img)
       
        
        after = len(web_direct[coco_class])
        assert before != after 
    else:
        done=True 
    return chosen_img, web_direct,done 
def create_full_data_3():
    total_data = []
    for i,coco_class in enumerate(UNSEEN_COMBINED):
        for web_c in coco_cat_to_web[coco_class]:
            for web_img in web_cat_to_image_id[web_c]:
                total_data.append((coco_class,web_img))
                #coco_class_to_image[coco_class].append(web_img)
    io.dump_json_object(total_data,'/shared/rsaas/michal5/gpv_michal/lessons/full_localization_data_3.json')
    return total_data

def create_full_data_2():
    total_num = 0
    all_data = []
    global_list = []
    classes_to_remove = []
    empty_classes = []



    coco_class_to_image,smallest_class = construct_unseen_to_web_direct(classes_to_remove)
    for i in coco_class_to_image:
        print(len(coco_class_to_image[i]))

    #find smallest class
    #add class to every entry in all data
    #perform 9 iterations 
    #add smallest class to classes to remove and empty classes 
    #create new unseen to web 
    while len(coco_class_to_image.keys())>1:
        new_data = []
        new_final_data = []
        for img in coco_class_to_image[smallest_class]:
            new_data.append({smallest_class:img})
        for i in range(9):
            for entry in new_data:

                c = choose_coco_class(list(entry.keys()),empty_classes)
                if c== None:
                    break
                if len(coco_class_to_image[c]) != 0:

                    chosen_img, coco_class_to_image,class_done = choose_image(c,coco_class_to_image)
                    if class_done:
                        del coco_class_to_image[c]
                        empty_classes.append(c)
                    entry[c] = chosen_img
                else:
                    del coco_class_to_image[c]
                    empty_classes.append(c)
            if len(list(entry.keys())) != len(set(entry.keys())):
                raise ValueError
        for entry in new_data:
            if len(entry) >= 5:
                global_list.append(entry)
                new_final_data.append(entry)
        classes_to_remove.append(smallest_class)
        empty_classes = [c for c in classes_to_remove]
        coco_class_to_image,smallest_class = construct_unseen_to_web_direct(classes_to_remove)
        #io.dump_json_object(new_final_data,f'/shared/rsaas/michal5/gpv_michal/full_localization_data_new_{len(coco_class_to_image.keys())}.json')
        #print(len(new_final_data),'all data')
        total_num += len(new_final_data)
        print(total_num,'new')
    if len(global_list) != total_num:
        raise ValueError 
    io.dump_json_object(global_list,f'/shared/rsaas/michal5/gpv_michal/lessons/full_localization_data_new_2.json')
        
    


        

    # coco_class_to_image_sorted = {k: v for k, v in sorted(coco_class_to_image.items(), key=lambda item: item[1])}
    # empty_classes = []
    # for i,k in enumerate(coco_class_to_image_sorted.keys()):

    #     if i ==0:
    #         classes_to_remove.append(k)
    #         for img in coco_class_to_image_sorted[k]:
    #             all_data.append({k:img})
        
    # for i in range(9):
    #     for entry in all_data:
    #         c = choose_coco_class(list(entry.keys()),empty_classes)
    #         if c== None:
    #             break
    #         if len(coco_class_to_image[c]) != 0:

    #             chosen_img, coco_class_to_image,class_done = choose_image(c,coco_class_to_image)
    #             if class_done:
    #                 del coco_class_to_image[c]
    #                 empty_classes.append(c)
    #             entry[c] = chosen_img
    #         else:
    #             del coco_class_to_image[c]
    #             empty_classes.append(c)
    # io.dump_json_object(all_data,'/shared/rsaas/michal5/gpv_michal/full_localization_data_1.json')
    # coco_class_to_image =  construct_unseen_to_web_direct(classes_to_remove)

    # all_data_2  = []
    # empty_classes = [classes_to_remove[0]]
    # coco_class_to_image_sorted_2 = {k: v for k, v in sorted(coco_class_to_image.items(), key=lambda item: item[1])}
    # if len(coco_class_to_image.keys()) >= 10:
    #     for img in coco_class_to_image_sorted_2[list(coco_class_to_image_sorted_2.keys())[0]]:
    #         all_data_2.append({list(coco_class_to_image_sorted_2.keys())[0]:img})
    # for i in range(9):
    #     for entry in all_data_2:
    #         c = choose_coco_class(list(entry.keys()),empty_classes)
    #         if c== None:
    #             break
    #         if len(coco_class_to_image[c]) != 0:

    #             chosen_img, coco_class_to_image,done = choose_image(c,coco_class_to_image)
    #             if done:
    #                 del coco_class_to_image[c]
    #                 empty_classes.append(c)

    #             entry[c] = chosen_img
    #         else:
    #             del coco_class_to_image[c]
    #             empty_classes.append(c)

        
            


    
        
     




 

def create_full_data():
    all_data = []
    same_length =0 
    last_num_same_length = 0
    coco_class_to_image = construct_unseen_to_web_direct()
    img_tracker = create_img_dict(coco_class_to_image)
    
    coco_class_iterators = construct_coco_class_iterators(coco_class_to_image)
    p = iter(UNSEEN_COMBINED)
    last_val = -1
    print(list(p))
    while len(all_data)<4000:
    # while  len(UNSEEN_COMBINED)> 10:
    #     if same_length >10:
    #         break
    #     last_num_same_length = len(all_data)
    #     if len(coco_class_iterators) < 10:
    #         break
        start_index = 0
    #     last_start = -1
    #     same_start = 0
        current_batch = {}
        while start_index < 10:
    #         if same_start >5:
    #             break
    #         last_num_same_length = len(all_data)
    #         if same_length >10:
    #             break
            try:
                current_class= next(p)
            except StopIteration:
                random.shuffle(UNSEEN_COMBINED)
                p = iter(UNSEEN_COMBINED)
                current_class = next(p)
    #         print(len(coco_class_iterators),'iterators')
    #         print(same_length,'same length')
    #         print(len(UNSEEN_COMBINED),current_class,len(all_data))
            if current_class in coco_class_iterators:
                img, coco_class_iterators = get_next_image(coco_class_to_image,coco_class_iterators,current_class)
    #         # print(current_batch,start_index)
    #         # print(current_class,'cu[rrent class')
                if current_class in current_batch.keys():
    #                 print('current class in batch',len(current_batch))
                     start_index = 10
                     continue
                if img_tracker[img] == 'True':
                     continue
                
                
    #             #assert current_class not in current_batch.keys()
                current_batch[current_class] = img 
                
                start_index += 1
    #             same_start = start_index
    #             if len(current_batch) ==9:
    #                 start_index = 10
    #             #print(img)
                img_tracker[img] = 'True' 
                coco_class_to_image[current_class].remove(img)
                if len(coco_class_to_image[current_class]) == 0:
                     del coco_class_to_image[current_class]
                     del coco_class_iterators[current_class]
                     UNSEEN_COMBINED.remove(current_class)
                     p = iter(UNSEEN_COMBINED)
                else:
                    coco_class_iterators[current_class] = iter(coco_class_to_image[current_class])
    #             if len(coco_class_iterators) <10:
    #                 break
    #         if start_index == last_val:
    #               same_start +=  1

            print(len(all_data),'all data')
        all_data.append(current_batch)
        
    #     if len(all_data) ==   last_val:
    #         same_length += 1 
    #     last_val = len(all_data)
    io.dump_json_object(all_data,'/shared/rsaas/michal5/gpv_michal/full_localization_data.json')
if __name__ == '__main__':
    # make_coco_cat_to_web_cat()
    # coco_cat_to_web = io.load_json_object('/data/michal5/web_training_info/coco_cat_to_web_cat_seen.json')

    #create_full_data_2()
    create_full_data_3()

        









