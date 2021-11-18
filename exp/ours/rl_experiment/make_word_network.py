from pyvis.network import Network
import os 
import nltk
import utils.io as io
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import networkx as nx
from tqdm import tqdm
lemmatizer = WordNetLemmatizer()
net = nx.Graph()
def load_json_information():
    web_qa_id_to_value = io.load_json_object('/home/michal/gpv_michal/query_to_web_id.json')
    category_names = [k for k in web_qa_id_to_value.keys()]
    category_to_coco_id = io.load_json_object('/data/michal5/web_training_info/category_coco_id.json')
    category_to_image_id = io.load_json_object('/data/michal5/web_training_info/category_to_image_id.json')
    category_to_pos = io.load_json_object('/data/michal5/web_training_info/category_to_pos.json')
    dict_list = {'category_names':category_names,'category_to_coco_id':category_to_coco_id,'category_to_image_id':category_to_image_id,
    'category_to_pos':category_to_pos,'web_id_to_value':web_qa_id_to_value}
    return dict_list 
dict_list = load_json_information()
dict_list['cat_to_num'] = io.load_json_object('/data/michal5/web_training_info/cat_to_num_id.json')
total = 0
no_wordnet = 0
adj_dict = {}
verb_dict = {}
noun_dict = {}
for cat in dict_list['cat_to_num']:
    net_id = dict_list['cat_to_num'][cat]
  
    category = cat
    if category in dict_list['category_to_pos']:

        pos = dict_list['category_to_pos'][category]

        coco_id = dict_list['category_to_coco_id'][category]
        image_ids = dict_list['category_to_image_id'][category]
        lemma_adj = None
        lemma_noun = None
        lemma_verb = None
        if pos['adj'] != None:
            lemma_adj = lemmatizer.lemmatize(pos['adj'],'a')
            if lemma_adj not in adj_dict:
                adj_dict[lemma_adj] = [net_id]
            else:
                adj_dict[lemma_adj].append(net_id)
        if pos['verb'] != None:
            lemma_verb = lemmatizer.lemmatize(pos['verb'],'v')
            if lemma_verb not in verb_dict:
                verb_dict[lemma_verb] = [net_id]
            else:
                verb_dict[lemma_verb].append(net_id)
        if pos['noun'] != None:
            lemma_noun = lemmatizer.lemmatize(pos['noun'],'n')
            if lemma_noun not in noun_dict:
                noun_dict[lemma_noun] = [net_id]
            else:
                noun_dict[lemma_noun].append(net_id)
        hypo_nyms = set()
        hyper_nyms = set()
        try:
            word = wordnet.synsets(lemma_noun)[0]
            for h in word.hypernyms():
                for l in h.lemmas():
                    hyper_nyms.add(l.name())
          
            for h in word.hyponyms():
                for l in h.lemmas():
                    hypo_nyms.add(l.name())
            total += 1
            net.add_node(net_id,hypernyms=hyper_nyms,hyponyms=hypo_nyms,label=category,pos=pos,image_ids=image_ids,title=category,coco_id=coco_id,lemma_adj=lemma_adj,lemma_verb=lemma_verb,lemma_noun=lemma_noun)
    
        except IndexError:
            #print(f'{lemma_noun} has no wordnets')
            total += 1
            no_wordnet += 1
            net.add_node(net_id,hypernyms=hyper_nyms,hyponyms=hypo_nyms,label=category,pos=pos,image_ids=image_ids,title=category,coco_id=coco_id,lemma_adj=lemma_adj,lemma_verb=lemma_verb,lemma_noun=lemma_noun)
    
            continue 
        
    else:
      
        continue
print(f'total:{total}')
print(f'no wordnet:{no_wordnet}')
for entry in tqdm(adj_dict):
    for v1 in tqdm(adj_dict[entry]):
        for v2 in tqdm(adj_dict[entry]):
            if v1 != v2:
         
                net.add_edge(int(v1),int(v2),color='blue')
                



for entry in tqdm(verb_dict):
    for v1 in tqdm(verb_dict[entry]):
        for v2 in tqdm(verb_dict[entry]):
            if v1 != v2:
                net.add_edge(v1,v2,color='green')
for entry in tqdm(noun_dict):
    for v1 in tqdm(noun_dict[entry]):
        for v2 in tqdm(noun_dict[entry]):
            net.add_edge(v1,v2,edge='red')
nx.write_gpickle(net, "/home/michal/gpv_michal/word_graph.gpickle")
# information_store = {}
# node_list_1 = net.get_nodes()
# node_list_2 = net.get_nodes()
# for n_1 in tqdm(node_list_1):
#     for n_2 in tqdm(node_list_2):
#         n_1_attributes = net.get_node(n_1)
#         n_2_attributes = net.get_node(n_2)
#         if n_1 not in information_store:
#             information_store[n_1] = {"adj_connections":[],"verb_connections":[],"noun_connections":[]}
#         if n_1 != n_2:
#             if n_1_attributes['lemma_adj'] != None and n_2_attributes['lemma_adj'] != None:
#                 if n_1_attributes['lemma_adj'] == n_2_attributes['lemma_adj']:
#                     net.add_edge(n_1,n_2,color='blue')
#                     information_store[n_1]["adj_connections"].append(n_2)
#             if n_1_attributes['lemma_verb'] != None and n_2_attributes['lemma_verb']!= None:
#                 if n_1_attributes['lemma_verb'] == n_2_attributes['lemma_verb']:
#                     net.add_edge(n_1,n_2,color='green')
#                     information_store[n_1]["verb_connections"].append(n_2)
#             if n_1_attributes['lemma_noun'] != None and n_2_attributes['lemma_noun'] != None:
#                 if n_1_attributes['lemma_noun'] == n_1_attributes['lemma_noun']:
#                     net.add_edge(n_1,n_2,color='red')
#                     information_store[n_1]["noun_connections"].append(n_2)
#                 else:
#                     if n_2_attributes['lemma_noun'] in n_1_attributes['hypernyms']:
#                         net.add_edge(n_1,n_2,color='red')
#                         information_store[n_1]["noun_connections"].append(n_2)
#                     elif n_2_attributes['lemma_noun'] in n_1_attributes['hyponyms']:
#                         net.add_edge(n_1,n_2,color='red')
#                         information_store[n_1]["noun_connections"].append(n_2)
       
#         io.dump_json_object(information_store,'/home/michal/gpv_michal/web_search_connections.json')
