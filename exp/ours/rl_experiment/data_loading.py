import json 
import logging 
import utils.io as io 
from nltk.corpus import wordnet
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.utils import make_grid,save_image ,draw_bounding_boxes
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



#step 0
#get corresponding categories and contrast categories 
#for now goal is to learn coarse so blue bed and red bed are both examples of bed

#step 1
#for each category 
# get list of image ids 
#get noun, verb, adjective 
#if mapped to coco category, can use super cateogry 
#if noun only, get category
#for adjectives, get type of adjective 

#for each category
#convert templates to questions and save under category 
#for each category, for each image, add question 
#get all questions. make sure each 

#step 2
#generate questions and corresponding answers

#step 3
#put questions and answers in form of 


from nltk.corpus import wordnet as wn
class Question():
    def __init__(self,question,answer):
        self.question = question
        self.answer = answer 
    
TEMPLATES = {}

TEMPLATES['adj'] = [
  "WH ADJ_TYPE is DT_OBJ?",
  "What is the ADJ_TYPE of DT_OBJ?",
  "CMD the ADJ_TYPE of DT_OBJ.",
]

TEMPLATES['noun_category'] = [
    'WH NOUN_CAT is this',
    'WH type of NOUN_CAT is this',
    'WH kind of NOUN_CAT is this',
    'WH class of NOUN_CAT is this'
]

TEMPLATES['verb'] = [
  'What is being done?',
  "WH action is being done?",
  "WH activity is being done?",
  "WH activity is this?",
  "WH action is being taken?",
  "CMD the activity being doing.",
  "CMD the action being doing.",
  "CMD the action being taken.",
]


TEMPLATES['verb_object'] = [
  "What is DT_OBJ doing?",
  "What action is DT_OBJ taking?",
  "What action is DT_OBJ performing?",
  "What action is DT_OBJ carrying out?",
  "What action is DT_OBJ doing?",
  "What activity is DT_OBJ doing?",
  "CMD the action being taken by DT_OBJ.",
  "CMD the activity DT_OBJ is doing.",
  "CMD what DT_OBJ is doing.",
]


TEMPLATES['query'] = [
  "What is this?",
  "What is that?",
]


TEMPLATES['noun'] = [
  "What is DT_OBJ?",
  "What OBJ is this?",
  "What OBJ is that?",
  "NAME DT_OBJ.",
]

SUBSTITUIONS = {
  "DT_OBJ": [
    "this object", "this entity", "this thing",
    "the object", "the entity",
    "that object", "that entity",  "that thing"
  ],
  "DT": ["the", "this", "that"],
  "OBJ": ['object', 'entity'],
  "CMD": ["Describe", "State", "Specify", "Name"],
  "NAME": ["Describe", "Specify", "Name", "Classify"],
  "CAP": ["Describe", "Caption", "Generate a caption for"],
  "WH": ["What", "Which"]
}

ADJ_TYPES = io.load_json_object("/home/michal/gpv_michal/exp/ours/data/webqa_adj_types.json")


def grid(class_name):
  transforms = torch.nn.Sequential(
    T.Resize((224,224))
)
  print(class_name)
  json_info = load_json_information()
  actual_contrast_classes = set()
 
  contrast_classes,first_order_classes = find_webqa_classes(json_info['category_names'],class_name)
  
  for c in contrast_classes:
    if c in json_info['category_to_image_id'].keys():
      actual_contrast_classes.add(c)
  print(len(actual_contrast_classes),'actual contrast classes')
  print(len(first_order_classes),'first_order_classes')
  actual_class = np.random.choice(list(first_order_classes),1)
  other_classes = np.random.choice(list(actual_contrast_classes),3)
  actual_image = np.random.choice(json_info['category_to_image_id'][actual_class.tolist()[0]],1)
  other_images = []
  #print(json_info['category_to_image_id'].keys())
  for cat in other_classes.tolist():
    print(cat,'cat')
    img_loc = np.random.choice(json_info['category_to_image_id'][cat],1)
    other_images.append(img_loc.tolist()[0])
  actual_img = read_image('/data/michal5/data/michal5/web_data/images/'+actual_image.tolist()[0])
  img_1 = read_image('/data/michal5/data/michal5/web_data/images/'+other_images[0])
  img_2 = read_image('/data/michal5/data/michal5/web_data/images/'+other_images[1])
  img_3 = read_image('/data/michal5/data/michal5/web_data/images/'+other_images[2])
  save_image(img_1.float()/255.0,'img_1_test.png')
  actual_img = transforms(actual_img)
  img_1 = transforms(img_1)
  img_2 = transforms(img_2)
  img_3 = transforms(img_3)
  grid = make_grid([actual_img,img_1,img_2,img_3])
  save_image(grid.float()/255.0,'test_img.jpg')
  boxes = torch.tensor([[0, 0, 224, 224], [224, 0, 448, 448],[448,0,672,672],[672,0,896,896]], dtype=torch.float)
  colors=['red','red','red','red']
  result = draw_bounding_boxes(grid, boxes,colors=colors, width=1)
  save_image(result.float()/255.0,'test_boxes.jpg')

def _expand_templates(templates):
  for (prefix, subs) in SUBSTITUIONS.items():
    out = []
    for template in templates:
      if prefix in template:
        for sub in subs:
          out.append(template.replace(prefix, sub))
      else:
        out.append(template)
    templates = out
  return templates

def expand_noun_cat(noun_cat):
    out = []
    noun_cat_templates = TEMPLATES['noun_category']
    for wh in SUBSTITUTIONS["WH"]:
        for noun in noun_cat_templates:
            temp_temp = template.replace("WH",wh)
            out.append(temp_temp.replace("NOUN_CAT",noun_cat))
    return out 

def _substitute_noun(templates, noun):
  out = []
  for x in templates:
    print('temp:',x)
    if "DT_OBJ" in x:
      temp = x.replace("DT_OBJ",f"DT {noun}")
      for t in SUBSTITUIONS["DT"]:
        out.append(temp.replace("DT",t))

    elif "OBJ" in x:
      out.append(x.replace("OBJ", noun))
    else:
      raise ValueError()
  return out


def get_noun_templates():
  return _expand_templates(TEMPLATES['noun'])


def get_query_tempates():
  return _expand_templates(TEMPLATES['query'])


def get_adj_templates(adj_type, noun=None):
  templates = [x.replace("ADJ_TYPE", adj_type) for x in TEMPLATES['adj']]
  if noun is not None:
    templates = _substitute_noun(templates, noun)
  return templates


def get_verb_templates(noun=None):
  templates = TEMPLATES['verb_object']
  if noun is not None:
    templates = _substitute_noun(templates, noun)
  else:
    # Add generic no-noun templates
    templates = TEMPLATES["verb"] + templates
  return templates
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


def get_hypernym(class_name):

  syn = wordnet.synsets(class_name)
  if len(syn) != 0:
    hypernyms = wordnet.synset(syn[0]).hypernyms()
    return hypernyms[0].name()
  else:
    return None




def find_webqa_classes(key_names,class_name):

    final_contrast_classes = set()
    first_order_classes = set()
    for k in key_names:
        if class_name in k:
            first_order_classes.add(k)
   
    contrast_classes = find_contrast_classes(class_name,first_order_classes)
    print(first_order_classes)

    for k in key_names:
      for c in contrast_classes:

        if c in k:
          if k not in first_order_classes:
            final_contrast_classes.add(k)
    
 
    return final_contrast_classes, first_order_classes
def generate_questions(category,json_info):
  
    #for c in first_order_classes:
    pos = json_info['category_to_pos'][category]
    print(pos)
    if pos['noun'] != None:
        noun_templates = get_noun_templates()
      
        #noun_questions = _substitute_noun(noun_templates, pos['noun'])
    else:
        noun_templates = None

    if pos['verb'] != None:
        if pos['noun'] != None:
            verb_templates = _expand_templates(get_verb_templates(noun=pos['noun']))
        else:
            verb_templates = _expand_templates(get_verb_templates())
    else:
        verb_templates = None

    if pos['adj'] != None:
        adj_templates = _expand_templates(get_adj_templates(adj_type=ADJ_TYPES[pos['adj']], noun=pos['noun']))
    else:
        adj_templates =  None 
    if verb_templates == None and adj_templates == None:
        noun_category_templates = expand_noun_cat(get_hypernym(category))
    else:
        noun_category_templates = None
    question_dict = {'pos':pos,'noun_questions':noun_templates,'noun_category_questions':noun_category_templates,'verb_questions':verb_templates,'adj_questions':adj_templates}
    return question_dict
    
        
def get_concept_questions(class_name):
    questions = {}
    json_info = load_json_information()
    contrast_classes,first_order_classes = find_webqa_classes(json_info['category_names'],class_name)
    test_class = list(first_order_classes)[0]
    qs = generate_questions(test_class,json_info)
    print(qs)

def load_json_information():
    web_qa_id_to_value = io.load_json_object('/home/michal/gpv_michal/query_to_web_id.json')
    category_names = [k for k in web_qa_id_to_value.keys()]
    category_to_coco_id = io.load_json_object('/home/michal/gpv_michal/exp/ours/web_training_info/category_coco_id.json')
    category_to_image_id = io.load_json_object('/home/michal/gpv_michal/exp/ours/web_training_info/category_to_image_id.json')
    category_to_pos = io.load_json_object('/home/michal/gpv_michal/exp/ours/web_training_info/category_to_pos.json')
    dict_list = {'category_names':category_names,'category_to_coco_id':category_to_coco_id,'category_to_image_id':category_to_image_id,
    'category_to_pos':category_to_pos}
    return dict_list 


if __name__ == '__main__':
  grid('bed')