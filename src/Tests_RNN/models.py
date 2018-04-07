import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from hyperparameters import *

# inspired from pytorch examples

#note by note: input size of 1 (one sample of input, i.e one note or one frequency)
class basic_rnn_discriminator(nn.Module):
    def __init__(self, HIDDEN_SIZE):
        super(basic_rnn_discriminator, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.loss =  nn.BCELoss()

        self.f_hidden = nn.Linear(SONG_PIECE_SIZE + HIDDEN_SIZE, HIDDEN_SIZE) # One layer only
        self.f_output = nn.Linear(SONG_PIECE_SIZE + HIDDEN_SIZE, 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.f_hidden(combined)
        output = self.f_output(combined)
        output = F.sigmoid(output)
        return output, hidden

    def forward_D(self, batch, current_batch_size):
        prediction = torch.zeros(current_batch_size)
        hidden = self.initHidden(current_batch_size)
        if cuda:
            hidden.cuda

        for song_piece_begin in range(0, batch.shape[1], SONG_PIECE_SIZE):
            res, hidden_res = self.forward(batch[:, song_piece_begin:song_piece_begin+SONG_PIECE_SIZE].float(), hidden)
            hidden = hidden_res
            prediction = res.data[:,0]
        return prediction


    def initHidden(self,batch_size):
        return Variable(torch.zeros(batch_size, self.HIDDEN_SIZE)) # h_0

#1 generator cell procuce one line of the resulting  song (temporary)
class basic_rnn_generator(nn.Module):
    def __init__(self, HIDDEN_SIZE, output_size):
        super(basic_rnn_generator, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.loss =  nn.BCELoss()

        self.f_hidden = nn.Linear(LATENT_DIMENSION + HIDDEN_SIZE, HIDDEN_SIZE)
        self.f_output = nn.Linear(LATENT_DIMENSION + HIDDEN_SIZE, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.f_hidden(combined)
        output = self.f_output(combined)
        output = self.softmax(output)
        return output, hidden

    def forward_G(self, current_batch_size, z):
        generated_batch = torch.FloatTensor()

        # generate song with the generator RNN
        hidden = self.initHidden(current_batch_size)
        if cuda:
            hidden.cuda()

        generated_batch = torch.zeros(current_batch_size, SONG_LENGTH)

        generated_batch_tmp, hidden = self.forward(z, hidden)
        generated_batch[:, 0:SONG_PIECE_SIZE] = generated_batch_tmp.data
        zeros = Variable(torch.zeros(current_batch_size, LATENT_DIMENSION).type(torch.FloatTensor))
        if cuda:
            zeros.cuda()

        for i in range(SONG_PIECE_SIZE, SONG_LENGTH, SONG_PIECE_SIZE):
            generated_batch_tmp, hidden = self.forward(zeros, hidden)
            generated_batch[:, i:i+SONG_PIECE_SIZE] = generated_batch_tmp.data

        return generated_batch

    def initHidden(self,batch_size):
        return Variable(torch.zeros(batch_size, self.HIDDEN_SIZE))
