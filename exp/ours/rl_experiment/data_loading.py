import json 
import logging 
import utils.io as io 

HELD_OUT_TRAINING_CLASSES = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass']
HELD_OUT_TEST_CLASSES = ['banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']

#list of tasks to generate 
#image+query =
#if noun only:
#Get questions from templates 
#For attributes
#get the type to create questions
#for localization, 
#find way to merge images from related classes 
#related classes should share other classes 
web_qa_id_to_value = io.load_json_object('/shared/rsaas/michal5/gpv_michal/query_to_web_id.json')
key_names = [k for k in web_qa_id_to_value.keys()]


from nltk.corpus import wordnet as wn
def _recurse_all_hypernyms(synset, all_hypernyms):
    synset_hypernyms = synset.hypernyms()
    if synset_hypernyms:
        all_hypernyms += synset_hypernyms
        for hypernym in synset_hypernyms:
            _recurse_all_hypernyms(hypernym, all_hypernyms)

def all_hypernyms(synset):
    hypernyms = []
    _recurse_all_hypernyms(synset, hypernyms)
    return set(hypernyms)

def check_person(word):
    if ' ' in word:
        word = word.split()[-1]
    flag = False
    if len(wn.synsets(word)) == 0:
        return False
    w = wn.synsets(word)[0]
    #for w in wn.synsets(word):
    if wn.synset('person.n.01') in all_hypernyms(w):
        flag = True
    return flag
def find_contrast_classes(class_name,first_order_classes):
    new_classes = set()
    for f in first_order_classes:
        if " " not in f:
            new_str = f.replace(class_name,"")
        else:
            new_str = find_word_with_class(class_name,f)
        new_classes.add(new_str)
    return new_classes

        
    #if a single word, replace with empty string 
    
    #if multi-word, remove word from string 
def find_word_with_class(class_name,first_order_class):
    indicies_to_remove = -1
    word_to_remove = class_name
    word_list = first_order_class.split(" ")
  
    for i,w in enumerate(word_list):
        if class_name in w:
        
            word_to_remove = w
    
    word_list.remove(word_to_remove)

    new_str = ""
    for i,w in enumerate(word_list):
        new_str+=w
        if len(word_list) > 1:
            if i<len(word_list)-1:
                new_str += " "

    return new_str

def get_domains(class_name):
    #get synset of domains
    #check if the hypernms are in domain
            

def find_webqa_classes(class_name):

    contrast_classes = set()
    first_order_classes = set()
    for k in key_names:
        if class_name in k:
            first_order_classes.add(k)
    print(len(first_order_classes))
    contrast_classes = find_contrast_classes(class_name,first_order_classes)
    
    print(contrast_classes)
    return contrast_classes, first_order_classes
def generate_questions(class_name)
find_webqa_classes('bed')
#find_word_with_class('bed','double bedding')