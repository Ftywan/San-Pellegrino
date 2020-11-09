# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
		# use torch library to save model parameters, hyperparameters, etc. to model_file

    # parse train file
    words, tags = [], []
    with open(train_file) as data:
      for line in data:
        line_words, line_tags = [], []
        line = line.strip()
        wordsntags = line.split(' ')
        for wordntag in wordsntags:
          line_tags.append(get_tag(wordntag))
          line_words.append(get_word(wordntag))
        words.append(line_words)
        tags.append(line_tags)
    print(words[0])
    print(tags[0])
    print(len(words), len(tags))

    # build word-to-ix, tag-to-ix
    



def get_tag(word):
    return word.rsplit('/', 1)[1]

def get_word(word):
    return word.rsplit('/', 1)[0].lower()

		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
