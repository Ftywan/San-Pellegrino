# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>
import datetime
start_time = datetime.datetime.now()

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# MODEL_CONSTANT
WORD_FEATURE_NUMBER = 128
CHAR_FEATURE_NUMBER = 32
CNN_KERNEL = 3
CNN_FILTERS_L = 32
LSTM_HIDDEN_SIZE = 64
LSTM_LAYER_NUMBER = 1
LSTM_DROPOUT = 0
EPOCHES = 3
LEARNING_RATE = 0.0015

# PROGRAM_CONSTANT
MIN_TIMEUP = 9
SEC_TIMEUP = 55

# global variable
device = torch.device("cuda:1" if torch.cuda.device_count()>1 else "cuda:0")
torch.manual_seed(7919)

def train_model(train_file, model_file):

    # parse train file
    words, tags = [], []
    word_index_map = {}
    tag_index_map = {}
    char_index_map = {}

    with open(train_file) as data:
        for line in data:
            line_words, line_tags = [], []
            line = line.strip()
            wordsntags = line.split(' ')
            for wordntag in wordsntags:
                word, tag = get_word_n_tag(wordntag)
                line_words.append(word)
                line_tags.append(tag)

                # to_inx
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
        tags[i] = [tag_index_map[tag] for tag in tags[i]] # convert all tag to index in tags

    num_lines = len(words)

    # train the model
    # losses = []
    loss_function = nn.CrossEntropyLoss()
    model = CNNBiLSTMModel(word_index_map, char_index_map, tag_index_map)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHES):
        for i in range(num_lines):
            # print("{}%".format(i/num_lines*100))
            sent_words = words[i]
            sent_tags = tags[i]
            model.zero_grad()

            output = model(sent_words).to(device)
            loss = loss_function(output, torch.LongTensor(sent_tags).to(device)).to(device)
            loss.backward()
            optimizer.step()

            # check time
            if (i+1) % 100 == 0:
                print(str(i/num_lines*100) + '%')
                time_diff = datetime.datetime.now() - start_time 
                if time_diff > datetime.timedelta(minutes=MIN_TIMEUP, seconds=SEC_TIMEUP):
                    terminate(word_index_map, char_index_map, tag_index_map, model.state_dict(), model_file)
                    return
        # losses.append(loss)
    terminate(word_index_map, char_index_map, tag_index_map, model.state_dict(), model_file)


def terminate(word_index_map, char_index_map, tag_index_map, state_dict, model_file):
    torch.save((word_index_map, char_index_map, tag_index_map, state_dict), model_file)
    print('seconds lapsed: {}'.format((datetime.datetime.now() - start_time).total_seconds()))


def get_word_n_tag(word):
    # return word.rsplit('/', 1)[0].lower(), word.rsplit('/', 1)[1]
    return word.rsplit('/', 1)[0], word.rsplit('/', 1)[1]


class CNNBiLSTMModel(nn.Module):

    def __init__(self, word_index_map, char_index_map, tag_index_map):
        super(CNNBiLSTMModel, self).__init__()

        self.word_index_map = word_index_map
        self.char_index_map = char_index_map
        self.num_words = len(word_index_map)
        self.num_chars = len(char_index_map)

        self.word_embeddings = nn.Embedding(self.num_words + 1, WORD_FEATURE_NUMBER, padding_idx=self.num_words).to(device)
        self.char_embeddings = nn.Embedding(self.num_chars + 1, CHAR_FEATURE_NUMBER, padding_idx=self.num_chars).to(device)

        self.conv1d = nn.Conv1d(in_channels=CHAR_FEATURE_NUMBER, out_channels=CNN_FILTERS_L,kernel_size=CNN_KERNEL, stride=1, padding=(CNN_KERNEL-1)//2, bias=True).to(device)
  
        self.lstm = nn.LSTM(
            input_size=WORD_FEATURE_NUMBER+CNN_FILTERS_L,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYER_NUMBER,
            dropout=LSTM_DROPOUT,
            bidirectional=True
        ).to(device)

        self.linear = nn.Linear(LSTM_HIDDEN_SIZE * 2, len(tag_index_map)).to(device)
        
        self.pool = nn.AdaptiveMaxPool1d(1).to(device)



    def forward(self, words):
        # feature engineering words
        word_tensor = torch.LongTensor([self.num_words if word not in self.word_index_map else self.word_index_map[word] for word in words]).to(device)
        
        word_embedding = self.word_embeddings(word_tensor).unsqueeze(1).to(device)

        # feature engineering chars
        char_tensor = [torch.LongTensor([self.num_chars if char not in self.char_index_map else self.char_index_map[char] for char in word]).to(device) for word in words]
        char_tensor = nn.utils.rnn.pad_sequence(char_tensor, batch_first=True, padding_value=self.num_chars)

        char_embedding = self.char_embeddings(char_tensor).transpose(1, 2).to(device)

        char_features = self.conv1d(char_embedding).to(device)
        char_features = self.pool(char_features).transpose(1, 2).to(device)

        # model input features
        features = torch.cat((word_embedding, char_features), dim=2).to(device)

        # prediction
        self.lstm.flatten_parameters()
        output, _ = self.lstm(features)
        result = self.linear(output.view(len(words), -1))
        return result


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
