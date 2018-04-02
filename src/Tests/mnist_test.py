# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models import *
from hyperparameters import *

import torchvision.datasets as dset
import torchvision.transforms as transforms

cuda = torch.cuda.is_available()

#global variable
index_list = np.arange(0, train_size)

#estimation
predictions = []

#Training
class Trainer():
    def __init__(self):
        self.D = D_conv()
        if cuda:
            self.D.cuda()
        self.G = G()
        if cuda:
            self.G.cuda()

        self.D_optimiser = optim.Adam(self.D.parameters(), lr = lr)
        self.G_optimiser = optim.Adam(self.G.parameters(), lr = lr)
        self.predictions = []

    def train(self, train_loader):
        global index_list
        for i in range(0, epoch):
            for k in range(0, D_steps):
                print("Epoch: " + str(i) + ", D step: " + str(k))
#                for batch_id in range(int((train_size - 1) / minibatch_size) + 1):
                for batch_idx, (x, target) in enumerate(train_loader):
                    #print(batch_idx)
                    self.D.zero_grad()

#                    size, data_batch = self.create_data_batch()
                    z = Variable(self.create_noise_batch())#.view(-1, G_inputs, 1, 1)

                    if cuda:
                        z = z.cuda()
                    generated_batch = self.G(z).detach()
                    x = Variable(x)
                    if cuda:
                        x = x.cuda()
                    real_prediction = self.D(x).squeeze() # 1x1

                    y_ones = Variable(torch.ones(minibatch_size))
                    if cuda:
                        y_ones = y_ones.cuda()
                    loss_d_r = self.D.loss(real_prediction, y_ones)
                    loss_d_r.backward()

                    generated_prediction = self.D(generated_batch.view(minibatch_size, 1, 28, 28)).squeeze() # 1x1
                    loss_d_f = self.D.loss(1 - generated_prediction, y_ones)
                    loss_d_f.backward()

                    self.D_optimiser.step()
                index_list = np.arange(0, train_size)

            for k in range(0, G_steps):
                z = Variable(self.create_noise_batch())#.view(-1, G_inputs, 1, 1))
                if cuda:
                    z = z.cuda()
                generated_batch = self.G(z)

                y_ones = Variable(torch.ones(minibatch_size))
                if cuda:
                    y_ones = y_ones.cuda()

                D_prediction = self.D(generated_batch)[0] # 1x1
                loss_G = self.G.loss(D_prediction, y_ones)
                loss_G.backward()

                predictions.append(D_prediction.mean().data)

                self.G_optimiser.step()


    def create_noise_batch(self):
        G_in = np.random.normal(0.0, 1.0, [minibatch_size, G_inputs])
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
            batch[i] = np.load("../data/" + str(names[i]) + ".npy")

        return size, torch.from_numpy(batch).type(torch.FloatTensor)


root = './dataMnist'
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=minibatch_size,
                shuffle=True)


T = Trainer()
T.train(train_loader)

plt.plot(predictions, label="test")
plt.show()
