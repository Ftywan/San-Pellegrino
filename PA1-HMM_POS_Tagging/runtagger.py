# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import json
from collections import defaultdict

def tag_sentence(test_file, model_file, out_file):
    lines = open(test_file).readlines()
    model = json.load(open(model_file))
    transition_propability = model['transition_probability']
    emission_probability = model['emission_probability']
    all_tags = list(emission_probability.keys())
    num_tags = len(all_tags)
    new_lines = []

    for tag in transition_propability:
        probs = defaultdict(lambda: 10e-15, transition_propability[tag])
        transition_propability[tag] = probs

    for tag in emission_probability:
        probs = defaultdict(lambda: 10e-15, emission_probability[tag])
        emission_probability[tag] = probs

    for line in lines:
        words = line.strip().split(' ')
        values = [[0 for i in range(num_tags)] for x in range(len(words))]
        prevs = [[0 for i in range(num_tags)] for x in range(len(words))]

        # first word
        for i in range(num_tags):
            tag = all_tags[i]
            word = words[0].lower()
            value = transition_propability['<s>'][tag] * emission_probability[tag][word]
            values[0][i] = value
            prevs[0][i] = '<s>'

        # next words
        for i in range(1, len(words)):
            word = words[i].lower()
            for j in range(num_tags):
                tag = all_tags[j]
                transitions = []
                for x in range(num_tags):
                    transitions.append(values[i-1][x] * transition_propability[all_tags[x]][tag])
                max_transition_value = max(transitions)
                prev_tag_index = transitions.index(max_transition_value)
                value = emission_probability[tag][word] * max_transition_value
                values[i][j] = value
                prevs[i][j] = prev_tag_index

        # ending state
        end_value = []
        for x in range(num_tags):
            end_value.append(values[-1][x] * transition_propability[all_tags[x]]['</s>'])
        final_tag_index = end_value.index(max(end_value))

        # traceback
        result_tags = []
        prev_index = final_tag_index
        for prev in reversed(prevs[1:]):
            result_tags.append(all_tags[prev_index])
            prev_index = prev[prev_index]
        result_tags.append(all_tags[prev_index])
        result_tags.reverse()
        
        # new_line_construction
        new_line = []
        for i in range(len(words)):
            tag = result_tags[i]
            new_word = words[i] + '/' + tag
            new_line.append(new_word)
        new_line = ' '.join(map(str, new_line)) + '\n'
        new_lines.append(new_line)
    
    # output
    with open(out_file, 'w') as out_file:
        for new_line in new_lines:
            out_file.write(new_line)


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
