# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WORD_VEC_DIM = 128
CHAR_VEC_DIM = 32
CNN_WINDOW_K = 3
CNN_FILTERS_L = 32
LSTM_FEATURES = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0

TIME_LIMIT_MIN = 9
TIME_LIMIT_SEC = 0
EPOCH = 2
LEARNING_RATE = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

word_index_map = {}
tag_index_map = {}
char_index_map = {}

def train_model(train_file, model_file):

    # parse train file
    words, tags = [], []
    with open(train_file) as data:
        for line in data:
            line_words, line_tags = [], []
            line = line.strip()
            wordsntags = line.split(' ')
            for wordntag in wordsntags:
                word, tag = get_word_n_tag(wordntag)
                line_words.append(word)
                line_tags.append(tag)
                if word not in word_index_map:
                    word_index_map[word] = len(word_index_map)
                if tag not in tag_index_map:
                    tag_index_map[tag] = len(tag_index_map)
                for char in word:
                    if char not in char_index_map:
                        char_index_map[char] = len(char_index_map)
            words.append(line_words)
            tags.append(line_tags)

    for i in range(len(tags)):
        tags[i] = [tag_index_map[tag] for tag in tags[i]]

    losses = []
    loss_function = nn.CrossEntropyLoss()
    model = CNNBiLSTMModel(word_index_map, char_index_map)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        loss = 0
        







def get_word_n_tag(word):
    return word.rsplit('/', 1)[0].lower(), word.rsplit('/', 1)[1]

class CNNBiLSTMModel(nn.Module):

    def __init__(self, word_vocab, char_vocab):
        super(CNNBiLSTMModel, self).__init__()

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.word_vocab_length = len(word_vocab)
        self.char_vocab_length = len(char_vocab)
        self.word_embeddings = nn.Embedding(self.word_vocab_length + 1, WORD_VEC_DIM, padding_idx=len(self.word_vocab)).to(device)
        self.char_embeddings = nn.Embedding(self.char_vocab_length + 1, CHAR_VEC_DIM, padding_idx=len(self.char_vocab)).to(device)

        self.conv1d = nn.Conv1d(in_channels=CHAR_VEC_DIM, out_channels=CNN_FILTERS_L,kernel_size=CNN_WINDOW_K, stride=1, padding=(CNN_WINDOW_K-1)//2, bias=True).to(device)
        self.pool = nn.AdaptiveMaxPool1d(1).to(device)

        self.lstm = nn.LSTM(
            input_size=WORD_VEC_DIM+CNN_FILTERS_L,
            hidden_size=LSTM_FEATURES,
            num_layers=LSTM_LAYERS,
            dropout=LSTM_DROPOUT,
            bidirectional=True
        ).to(device)

        self.hidden2tag = nn.Linear(
            LSTM_FEATURES * 2, len(tag_index_map)).to(device)

    def forward(self, word_sentence):

        word_input = torch.LongTensor(
            [self.word_vocab_length if word not in self.word_vocab else self.word_vocab[word] for word in word_sentence]).to(device)
        word_embeds = self.word_embeddings(word_input).unsqueeze(1).to(device)

        char_input = [torch.LongTensor([len(self.char_vocab) if char not in self.char_vocab else self.char_vocab[char]
                                        for char in word]).to(device) for word in word_sentence]

        char_input = nn.utils.rnn.pad_sequence(
            char_input, batch_first=True, padding_value=self.char_vocab_length)

        char_embeds = self.char_embeddings(char_input).transpose(1, 2).to(device)

        char_rep = self.conv1d(char_embeds).to(device)

        char_rep = self.pool(char_rep).transpose(1, 2).to(device)

        word_rep = torch.cat((word_embeds, char_rep), dim=2).to(device)

        self.lstm.flatten_parameters()
        output, _ = self.lstm(word_rep)
        out = self.hidden2tag(output.view(len(word_input), -1))
        return out


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
