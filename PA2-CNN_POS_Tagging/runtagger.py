# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch
from buildtagger import CNNBiLSTMModel
from collections import defaultdict

device = torch.device("cuda:1" if torch.cuda.device_count()>1 else "cuda:0")

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file
    word_index_map, char_index_map, tag_index_map, model_state = torch.load(model_file)
    word_index_map = defaultdict(lambda: len(word_index_map), word_index_map)
    char_index_map = defaultdict(lambda: len(char_index_map), char_index_map)
    index_tag_map = {index:tag for tag, index in tag_index_map.items()}
    model = CNNBiLSTMModel(word_index_map, char_index_map, tag_index_map)
    model.load_state_dict(model_state)

    outfile = open(out_file, 'w')
    with open(test_file) as data:
      for line in data:
        line = line.strip()
        words = line.split(' ')
        output = torch.argmax(model(words).to(device), dim=1).to(device)
        tags = [index_tag_map[index] for index in output.tolist()]

        result = []
        for i in range(len(words)):
          result.append(words[i] + '/' + tags[i])
        result_string = ' '.join(map(str, result)) + '\n'
        outfile.write(result_string)

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
