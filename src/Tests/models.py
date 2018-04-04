import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.loss = nn.BCELoss()
        self.fc1 = nn.Linear(image_x * image_y, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.loss = nn.BCELoss()
        self.fc1 = nn.Linear(G_inputs, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, image_y * image_x)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class D_conv(nn.Module):
    def __init__(self):
        super(D_conv, self).__init__()
        self.loss = nn.BCELoss()
        self.conv1 = nn.Conv2d(1, 4 * model_complexity, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4 * model_complexity)
        self.layer1 = nn.Sequential(self.conv1, self.bn1)
        self.conv2 = nn.Conv2d(4 * model_complexity, 8 * model_complexity, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8 * model_complexity)
        self.layer2 = nn.Sequential(self.conv2, self.bn2)
        self.conv3 = nn.Conv2d(8 * model_complexity, 16 * model_complexity, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16 * model_complexity)
        self.layer3 = nn.Sequential(self.conv3, self.bn3)
        self.fully = nn.Linear(16 * image_x * image_y * model_complexity, 1)
        self.weights_init(mean, std)

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

    def forward(self, x):
        out = F.leaky_relu(self.layer1(x), leak)
        out = F.leaky_relu(self.layer2(out), leak)
        out = F.leaky_relu(self.layer3(out), leak)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(self.fully(out))
        return out

class G_conv(nn.Module):
    def __init__(self):
        super(G_conv, self).__init__()
        self.loss = nn.BCELoss()
        self.conv1 = nn.ConvTranspose2d(G_inputs, 16 * model_complexity, 5, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16 * model_complexity)
        self.layer1 = nn.Sequential(self.conv1, self.bn1)
        self.conv2 = nn.ConvTranspose2d(16 * model_complexity, 8 * model_complexity, 5, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(8 * model_complexity)
        self.layer2 = nn.Sequential(self.conv2, self.bn2)
        self.conv3 = nn.ConvTranspose2d(8 * model_complexity, 4 * model_complexity, 4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(4 * model_complexity)
        self.layer3 = nn.Sequential(self.conv3, self.bn3)
        self.deconv4 = nn.ConvTranspose2d(4 * model_complexity, 1, 4, padding=1, stride=2)
        self.weights_init(mean, std)

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.tanh(self.deconv4(out))
        return out

def weights_init_general(model, mean, std):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(mean, std)
            model._modules[m].bias.data.zero_()

class basic_rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(basic_rnn, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

class basicLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(basicLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
