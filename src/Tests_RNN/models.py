import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from hyperparameters import *

#from pytorch documentation

#note by note: input size of 1 (one sample of input, i.e one note or one frequency)
class basic_rnn_discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(basic_rnn_discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.loss =  nn.BCELoss()

        self.f_hidden = nn.Linear(step + hidden_size, hidden_size) # One layer only
        self.f_output = nn.Linear(step + hidden_size, 1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), -1)
        hidden = self.f_hidden(combined)
        output = self.f_output(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size)) # h_0

#1 generator cell procuce one line of the resulting  song
class basic_rnn_generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(basic_rnn_generator, self).__init__()
        self.hidden_size = hidden_size
        self.loss =  nn.BCELoss()

        self.f_hidden = nn.Linear(1 + hidden_size, hidden_size)
        self.f_output = nn.Linear(1 + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), -1)
        hidden = self.f_hidden(combined)
        output = self.f_output(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
