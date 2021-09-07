# Generate questions of various templates, given image info.

# Not meant to be run -- I haven't included the dependency files. 
# Just to be used as a source of code for the templating.

import pdb
import json
import random
random.seed(24)

"""
Templates:

1a:
    “{What|Which} [adj_type] is {this|the} [noun]?”
    “What is the [adj_type] of {this|the} [noun]?”
    “{Describe|State|Characterize|Specify|Name} the [adj_type] of {this|the} [noun].”

1v:
    “What is {this|the} [noun] doing?”
    “What action is {this|the} [noun] taking?” (I don’t want to use the word “performing” instead of “taking” because that is an action in itself, e.g. dancer performing)
    “{Describe|State|Characterize|Specify|Name} the action being taken by {this|the} [noun].”

"""

TEMPLATES = {}
# Replace DT with "this/the", NOUN with "noun/object/entity/thing" 
# with equal probability
TEMPLATES['adj'] = [
        "What ADJ_TYPE is DT NOUN?", 
        "Which ADJ_TYPE is DT NOUN?", 
        "What is the ADJ_TYPE of DT NOUN?", 
        "Describe the ADJ_TYPE of DT NOUN.",
        "State the ADJ_TYPE of DT NOUN.", 
        "Characterize the ADJ_TYPE of DT NOUN.",
        "Specify the ADJ_TYPE of DT NOUN.",
        "Name the ADJ_TYPE of DT NOUN.", 
        ]
TEMPLATES['verb'] = [
        "What is DT NOUN doing?",
        "What action is DT NOUN taking?",
        "Describe the action being taken by DT NOUN.",
        "State the action being taken by DT NOUN.",
        "Characterize the action being taken by DT NOUN.",
        "Specify the action being taken by DT NOUN.",
        "Name the action being taken by DT NOUN.",
        ]
TEMPLATES['noun'] = [
        "What is this?",
        "What is DT object?",
        "What is DT entity?",
        "What is DT thing?",
        "What object is this?",
        "What entity is this?",
        "What thing is this?",
        ]

flipped_attributes = json.load(open('flipped_attributes.json'))

def add_noun_question(image_info, question_counter):
    selected_template = random.sample(TEMPLATES['noun'], 1)[0]
    if 'DT' in selected_template:
        if random.randint(1,2) == 1:
            selected_template = selected_template.replace('DT', 'the')
        else:
            selected_template = selected_template.replace('DT', 'this')
    image_info.update({
            'query': selected_template,
            'answer': image_info['noun'],
            'id': question_counter,
            'instance_id': question_counter,
        })
    return image_info


def add_adj_question(image_info, question_counter):
    selected_template = random.sample(TEMPLATES['adj'], 1)[0]
    adj_type = flipped_attributes[image_info['adj']]
    if 'ADJ_TYPE' in selected_template:
        selected_template = selected_template.replace('ADJ_TYPE', adj_type)
    if 'DT' in selected_template:
        if random.randint(1,2) == 1:
            selected_template = selected_template.replace('DT', 'the')
        else:
            selected_template = selected_template.replace('DT', 'this')
    if 'NOUN' in selected_template:
        random_no = random.randint(1,4)
        if random_no == 1:
            selected_template = selected_template.replace('NOUN', image_info['noun'])
        elif random_no == 2:
            selected_template = selected_template.replace('NOUN', 'object')
        elif random_no == 3:
            selected_template = selected_template.replace('NOUN', 'entity')
        elif random_no == 4:
            selected_template = selected_template.replace('NOUN', 'thing')
    image_info.update({
            'query': selected_template,
            'answer': image_info['adj'],
            'id': question_counter,
            'instance_id': question_counter,
        })
    return image_info


def add_verb_question(image_info, question_counter):
    selected_template = random.sample(TEMPLATES['verb'], 1)[0]
    if 'DT' in selected_template:
        if random.randint(1,2) == 1:
            selected_template = selected_template.replace('DT', 'the')
        else:
            selected_template = selected_template.replace('DT', 'this')
    if 'NOUN' in selected_template:
        random_no = random.randint(1,4)
        if random_no == 1:
            selected_template = selected_template.replace('NOUN', image_info['noun'])
        elif random_no == 2:
            selected_template = selected_template.replace('NOUN', 'object')
        elif random_no == 3:
            selected_template = selected_template.replace('NOUN', 'entity')
        elif random_no == 4:
            selected_template = selected_template.replace('NOUN', 'thing')
    image_info.update({
            'query': selected_template,
            'answer': image_info['verb'],
            'id': question_counter,
            'instance_id': question_counter,
        })
    return image_info


def main():
    splits = ['train', 'val', 'test', 'minval']
    qas = {s: [] for s in splits}
    question_counter = 0
    for split in splits:
        filename = '{}_image_info.json'.format(split)
        image_infos = json.load(open(filename))
        
        for image_info in image_infos:
            # If noun, pick a template and add a question
            if image_info['noun']:
                qas[split].append(add_noun_question(image_info, question_counter))
                question_counter += 1
            # If adj, "
            if image_info['adj']:
                qas[split].append(add_adj_question(image_info, question_counter))
                question_counter += 1
            # If verb, "
            if image_info['verb']:
                qas[split].append(add_verb_question(image_info, question_counter))
                question_counter += 1
    pdb.set_trace()
    print()


if __name__ == '__main__':
    main()


