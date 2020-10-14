# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>
import os
import math
import sys
import datetime
from collections import defaultdict
import json

START = '<s>'
END = '</s>'

def train_model(train_file, model_file):
    transitions = defaultdict(lambda: defaultdict(lambda: 0))
    emissions = defaultdict(lambda: defaultdict(lambda: 0))
    lines = open(train_file).readlines()
    # count
    for line in lines:
        line = line.strip()
        words = line.split(' ')

        # transition and emission counts
        transitions[START][get_tag(words[0])] += 1
        transitions[get_tag(words[-1])][END] += 1
        for i in range(len(words)-1):
            tag = get_tag(words[i])
            word = get_word(words[i])
            transitions[tag][get_tag(words[i+1])] += 1
            emissions[tag][word] += 1
        emissions[get_tag(words[-1])][get_word(words[-1])] += 1
        
    # probability
    transition_probability = {}
    emission_probability = {}

    for key in transitions:
        transition_probability[key] = {}
        val = transitions[key]
        count = sum(val.values())
        for next_tag in val:
            conditional_count = val[next_tag]
            transition_probability[key][next_tag] = conditional_count / count
    
    for tag in emissions:
        emission_probability[tag] = {}
        word_counts = emissions[tag]
        count = sum(word_counts.values())
        for word in word_counts:
            conditional_count = word_counts[word]
            emission_probability[tag][word] = conditional_count / count
    
    result = {
                'transition_probability': transition_probability, 
                'emission_probability': emission_probability
                }
        
    json.dump(result, open(model_file, 'w'))

def get_tag(word):
    return word.rsplit('/', 1)[1]

def get_word(word):
    return word.rsplit('/', 1)[0].lower()

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
