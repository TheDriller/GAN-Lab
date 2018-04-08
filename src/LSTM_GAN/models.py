import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from hyperparameters import *

# inspired from C-RNN-GAN
# https://arxiv.org/pdf/1611.09904.pdf

class LSTM_generator(nn.Module):
    def __init__(self, HIDDEN_SIZE, output_size):
        super(LSTM_generator, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE + LATENT_DIMENSION
        self.loss = nn.BCELoss()
        self.lstm = nn.LSTM(input_size = LATENT_DIMENSION, hidden_size = HIDDEN_SIZE + LATENT_DIMENSION, num_layers = LSTM_LAYERS, bias = True, batch_first = True, dropout = DROPOUT_PROB, bidirectional = False)
        self.final_layer = nn.Linear(self.HIDDEN_SIZE, SONG_PIECE_SIZE)

    def forward_G(self, current_batch_size, z): #maybe here if concatenated correctly all at one is good
        print("forward G")
        print(z.size())
        generated_batch = torch.zeros(current_batch_size, SONG_LENGTH)
        hidden = self.initHidden(current_batch_size)
        if cuda:
            hidden = hidden.cuda()
            
        generated_batch_temp = self.lstm.forward(z, hidden)[0][:,0,:]
        generated_batch[:,0:SONG_PIECE_SIZE] = self.final_layer(generated_batch_temp).data
        zeros = Variable(torch.zeros(current_batch_size, 1, LATENT_DIMENSION)).type(torch.FloatTensor)

        if cuda :
            zeros = zeros.cuda()

        for i in range(SONG_PIECE_SIZE, SONG_LENGTH, SONG_PIECE_SIZE):
            generated_batch_temp, hidden = self.lstm.forward(zeros, hidden)
            generated_batch_temp = self.final_layer(generated_batch_temp)
            generated_batch[:, i:i+SONG_PIECE_SIZE] = generated_batch_temp.data

        return generated_batch

    def initHidden(self, current_batch_size):
        return (Variable(torch.zeros(1, current_batch_size, self.HIDDEN_SIZE))
                ,Variable(torch.zeros(1, current_batch_size,  self.HIDDEN_SIZE))) # (h_0, c_0)

class LSTM_discriminator(nn.Module):
    def __init__(self, HIDDEN_SIZE):
        super(LSTM_discriminator, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE + SONG_PIECE_SIZE
        self.loss = nn.BCELoss()
        self.lstm = nn.LSTM(input_size = SONG_PIECE_SIZE, hidden_size = HIDDEN_SIZE + SONG_PIECE_SIZE, num_layers = LSTM_LAYERS, bias = True, batch_first = True, dropout = DROPOUT_PROB, bidirectional = True)
        self.final_layer = nn.Linear(self.HIDDEN_SIZE * 2, 1) # convert LSTM output to decision (bidirectional)

    def forward_D(self, input): # maybe a for in here
        hidden = self.initHidden()
        if cuda:
            hidden = hidden.cuda()

        print(self.HIDDEN_SIZE)
        print(input.size())
        print(hidden[0].size())
        output, hiddens = self.lstm(input, hidden) # run throught the LSTM
        print(output.size())
        output = F.sigmoid(self.final_layer(output[output.size(0)-1,:,:])) # convert the output to the wanted dimension
        return output

    def initHidden(self):
        return (Variable(torch.zeros(2, 13778, self.HIDDEN_SIZE)),#not sure why: maybe need alocation at beginning and fills afterwards all hidden outputs
                Variable(torch.zeros(2, 13778, self.HIDDEN_SIZE)))# (h_0, c_0)
