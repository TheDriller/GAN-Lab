# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models import *
from hyperparameters import *
import os

import torchvision.datasets as dset
import torchvision.transforms as transforms

cuda = torch.cuda.is_available()

predictions = []
g_loss = []
d_loss = []

def load_real_songs():
    directory_str = "data/npy/"
    directory = os.fsencode(directory_str)
    song_nb = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    songs = np.zeros((song_nb,SONG_LENGTH))
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            filepath = directory_str + filename
            songs[i,:] = np.load(filepath)
            i = i+1
    np.random.shuffle(songs)
    return songs

def get_targets(length):
    y_almost_ones = Variable(torch.from_numpy(np.random.uniform(0.8, 1.0, length)).type(torch.FloatTensor))
    if cuda:
        y_almost_ones = y_almost_ones.cuda()

    y_almost_zeros = Variable(torch.from_numpy(np.random.uniform(0.0, 0.2, length)).type(torch.FloatTensor))
    if cuda:
        y_almost_zeros = y_almost_zeros.cuda()
    return y_almost_ones,y_almost_zeros

#Training
class Trainer():
    def __init__(self):
        self.D = basic_rnn_discriminator(HIDDEN_SIZE)
        if cuda:
            self.D.cuda()
        self.G = basic_rnn_generator(HIDDEN_SIZE, SONG_PIECE_SIZE)
        if cuda:
            self.G.cuda()

        self.D_optimiser = optim.Adam(self.D.parameters(), lr = LR, betas = (BETA1, BETA2))
        self.G_optimiser = optim.Adam(self.G.parameters(), lr = LR, betas = (BETA1, BETA2))
        self.predictions = []


    def train_generator(self,batch):
        # Train generator
        current_batch_size = batch.shape[0]
        losses = []

        for k in range(0, G_STEPS): # maybe add / remove training over MINIBATCH_SIZE
            z = Variable(self.create_noise_batch(current_batch_size))
            if cuda:
                z = z.cuda()

            print("start G")
            generated_batch = Variable(self.forward_G(current_batch_size))
            print("generated")
            generated_prediction = Variable(self.forward_D(generated_batch.view(1, -1), current_batch_size))

            loss_G = self.G.loss(Variable(generated_prediction, requires_grad=True), y_almost_ones)
            temp.append(loss_G.data)
            loss_G.backward()

            predictions.append(generated_prediction.mean())
            last_g_loss = loss_G.data[0]
            losses.append(last_g_loss)
            if last_g_loss < 0.7 * last_d_loss:
                self.G_optimiser.step()
        return losses

    def train_discriminator(self,batch):
        # Train discriminator
        current_batch_size = batch.shape[0]

        losses = []
        y_almost_ones,y_almost_zeros = get_targets(current_batch_size)
        for k in range(0, D_STEPS):
            self.D.zero_grad()
            print("real prediction start")
            real_prediction = self.forward_D(batch, current_batch_size)
            print(real_prediction)
            print("generated batch start")
            generated_batch = Variable(self.forward_G(current_batch_size))

            loss_d_r = self.D.loss(Variable(real_prediction, requires_grad=True), y_almost_ones)

            print("generated prediction start")
            generated_prediction = self.forward_D(generated_batch.view(1, -1), current_batch_size)

            print(generated_prediction)
            loss_d_f = self.D.loss(Variable(generated_prediction, requires_grad=True), y_almost_zeros)
            loss_total = loss_d_r + loss_d_f
            loss_total.backward()

            losses.append(loss_d_r.data + loss_d_f.data)
            last_d_loss = (loss_d_r + loss_d_f).data[0]
            if last_d_loss > 0.7 * last_g_loss:
                self.D_optimiser.step()
        return losses



    def train(self,real_songs):
        # global index_list
        z_saved = Variable(self.create_noise_batch(1).view(1, LATENT_DIMENSION, 1, 1))
        last_d_loss = 0
        last_g_loss = 0

        for i in range(0, TOT_EPOCHS):
            print("TOT_EPOCHS: " + str(i))
            real_song_nb = real_songs.shape[0]
            print(int((real_song_nb- 1) / MINIBATCH_SIZE) + 1)
            for batch_id in range(0,real_song_nb,MINIBATCH_SIZE):
                print("New batch")
                batch = torch.from_numpy(real_songs[batch_id:batch_id+MINIBATCH_SIZE-1])
                batch = Variable(batch.double())
                if cuda:
                    batch = batch.cuda()
                current_batch_size = batch.shape[0]

                # Train discriminator
                discriminator_losses = self.train_discriminator(batch)
                d_loss.append(np.mean(discriminator_losses))

                print(current_batch_size + G_STEPS)

                # Train generator
                generator_losses = self.train_generator(batch)

                g_loss.append(np.mean(generator_losses))
                print("done G")
                print(g_loss)

    def create_noise_batch(self, batch_size):
        G_in = np.random.normal(0.0, 1.0, [batch_size, LATENT_DIMENSION])
        return torch.from_numpy(G_in).type(torch.FloatTensor)



    def forward_G(self, current_batch_size):
        z = Variable(self.create_noise_batch(current_batch_size))
        if cuda:
            z = z.cuda()

        generated_batch = torch.FloatTensor()

        # generate song with the generator RNN
        hidden = self.G.initHidden()
        for batch_element in range(0, current_batch_size):
            for song_piece_begin in range(0, LATENT_DIMENSION):
                generated_batch_tmp, hidden = self.G.forward(z[batch_element, song_piece_begin].view(1, 1), hidden)
                generated_batch = torch.cat((generated_batch, generated_batch_tmp.data[0,:]))
        print(generated_batch)
        return generated_batch

    def forward_D(self, batch, current_batch_size):
        prediction = torch.zeros(current_batch_size)
        hidden = self.D.initHidden()
        for batch_element in range(0, current_batch_size):
            for song_piece_begin in range(0, batch.shape[1], SONG_PIECE_SIZE):
                res, hidden_res = self.D.forward(batch[batch_element, song_piece_begin:song_piece_begin+SONG_PIECE_SIZE].view(1, SONG_PIECE_SIZE).float(), hidden)
                hidden = hidden_res
                prediction[batch_element] = res.data[0,0]
        print(prediction)
        return prediction

T = Trainer()
real_songs = load_real_songs()
T.train(real_songs)

print("done")

torch.save(T.G.state_dict(), "g_saved.pt")
torch.save(T.D.state_dict(), "d_saved.pt")

if save:
    plt.plot(predictions, label="test")
    plt.savefig("predictions.png")
    plt.show()

    plt.plot(d_loss, label="d_loss")
    plt.plot(g_loss, label="g_loss")
    plt.legend(loc="best")
    plt.savefig("loss.png")
    plt.show()
