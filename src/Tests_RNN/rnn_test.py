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

#global variable
index_list = np.arange(0, train_size)

#estimation
predictions = []
g_loss = []
d_loss = []

#Training
class Trainer():
    def __init__(self):
        self.D = basic_rnn_discriminator(hidden_size)
        if cuda:
            self.D.cuda()
        self.G = basic_rnn_generator(hidden_size, step)
        if cuda:
            self.G.cuda()

        self.D_optimiser = optim.Adam(self.D.parameters(), lr = lr, betas = (beta1, beta2))
        self.G_optimiser = optim.Adam(self.G.parameters(), lr = lr, betas = (beta1, beta2))
        self.predictions = []

    def train(self):
        global index_list
        z_saved = Variable(self.create_noise_batch(1).view(1, latent_dimension, 1, 1))
        last_d_loss = 0
        last_g_loss = 0

        for i in range(0, epoch):
            print("Epoch: " + str(i))

            print(int((train_size - 1) / minibatch_size) + 1)
            for batch_id in range(int((train_size - 1) / minibatch_size) + 1):
                print("back")
                size, x = self.create_data_batch()
                print(size)
                if size == 0:
                    break
                print(x)
                x = Variable(x)
                if cuda:
                    x = x.cuda()
                current_batch_size = x.shape[0]

                y_almost_ones = Variable(torch.from_numpy(np.random.uniform(0.8, 1.0, current_batch_size)).type(torch.FloatTensor))
                if cuda:
                    y_almost_ones = y_almost_ones.cuda()

                y_almost_zeros = Variable(torch.from_numpy(np.random.uniform(0.0, 0.2, current_batch_size)).type(torch.FloatTensor))
                if cuda:
                    y_almost_zeros = y_almost_zeros.cuda()

                temp = []
                for k in range(0, D_STEPS):
                    self.D.zero_grad()
                    print("real prediction start")
                    print(self.forward_D(x, current_batch_size))
                    print("generated batch start")
                    generated_batch = Variable(self.forward_G(current_batch_size))

                    loss_d_r = self.D.loss(Variable(real_prediction, requires_grad=True), y_almost_ones)
                    print(loss_d_r)
                    loss_d_r.backward()

                    print("generated prediction start")
                    generated_prediction = self.forward_D(generated_batch.view(1, -1), current_batch_size)

                    print(generated_prediction)
                    loss_d_f = self.D.loss(Variable(generated_prediction, requires_grad=True), y_almost_zeros)
                    print(loss_d_f)
                    temp.append(loss_d_r.data + loss_d_f.data)
                    loss_d_f.backward()
                    last_d_loss = (loss_d_r + loss_d_f).data[0]
                    if last_d_loss > 0.7 * last_g_loss:
                        self.D_optimiser.step()

                d_loss.append(np.mean(temp))
                temp_loc = []

                print(current_batch_size + G_steps)

                for k in range(0, G_steps): # maybe add / remove training over minibatch_size
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
                    temp_loc.append(last_g_loss)
                    if last_g_loss < 0.7 * last_d_loss:
                        self.G_optimiser.step()

                g_loss.append(np.mean(temp_loc))
                print("done G")
                print(g_loss)

            #image_temp = self.G.forward(z_saved).view(1, image_x, image_y)
            #os.makedirs("epoch_images", exist_ok = True)
            #image.imsave("epoch_images/" + str(i) + ".png", image_temp[0].data)

    def create_noise_batch(self, batch_size):
        G_in = np.random.normal(0.0, 1.0, [batch_size, latent_dimension])
        return torch.from_numpy(G_in).type(torch.FloatTensor)

    def create_data_batch(self):
        global index_list
        size = np.min([minibatch_size, index_list.shape[0]])

        if(size == 0):
            return 0, []

        indices = np.random.randint(0, index_list.shape[0], size = size)
        names = index_list[indices]
        index_list = np.delete(index_list, names)
        batch = np.zeros([size, image_x * image_y])

        for i in range (0, size):
            batch[i] = np.load("../../data/" + str(names[i]) + ".npy")

        return size, torch.from_numpy(batch).type(torch.FloatTensor)

    def forward_G(self, current_batch_size):
        z = Variable(self.create_noise_batch(current_batch_size))
        if cuda:
            z = z.cuda()

        generated_batch = torch.FloatTensor()

        # evaluate generator RNN, by creating an image
        hidden = self.G.initHidden()
        for rnn_i in range(0, current_batch_size):
            for rnn_j in range(0, latent_dimension):
                generated_batch_tmp, hidden = self.G.forward(z[rnn_i, rnn_j].view(1, 1), hidden)
                generated_batch = torch.cat((generated_batch, generated_batch_tmp.data[0,:]))
        print(generated_batch)
        return generated_batch

    def forward_D(self, x, current_batch_size):
        print(x.size())
        real_prediction = torch.zeros(current_batch_size)
        hidden = self.D.initHidden()
        for rnn_i in range(0, current_batch_size):
            for rnn_j in range(0, x.shape[1], step):
                #print(self.D.forward(x[rnn_i, rnn_j:rnn_j+step].view(1, step), hidden))
                res, hidden_res = self.D.forward(x[rnn_i, rnn_j:rnn_j+step].view(1, step), hidden)
                hidden = hidden_res
                real_prediction[rnn_i] = res.data[0,0]
        print(hidden)
        print(real_prediction)
        return real_prediction

T = Trainer()
T.train()

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
