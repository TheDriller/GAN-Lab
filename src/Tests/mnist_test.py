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
g_loss = []
d_loss = []

#Training
class Trainer():
    def __init__(self):
        self.D = D_conv()
        if cuda:
            self.D.cuda()
        self.G = G_conv()
        if cuda:
            self.G.cuda()

        self.D_optimiser = optim.Adam(self.D.parameters(), lr = lr, betas = (beta1, beta2))
        self.G_optimiser = optim.Adam(self.G.parameters(), lr = lr, betas = (beta1, beta2))
        self.predictions = []

    def train(self, train_loader):
        global index_list
        for i in range(0, epoch):
            print("Epoch: " + str(i))
            temp = []

            for batch_idx, (x, target) in enumerate(train_loader):
                if batch_idx > 20:
                    break

                x = Variable(x)
                if cuda:
                    x = x.cuda()

                current_batch_size = x.shape[0]

                y_almost_ones = Variable(torch.from_numpy(np.random.normal(0.8, 1.0, current_batch_size)).type(torch.FloatTensor))
                if cuda:
                    y_almost_ones = y_almost_ones.cuda()

                y_almost_zeros = Variable(torch.from_numpy(np.random.normal(0.0, 0.2, current_batch_size)).type(torch.FloatTensor))
                if cuda:
                    y_almost_zeros = y_almost_zeros.cuda()

                for k in range(0, D_steps):
                    self.D.zero_grad()

#                    size, data_batch = self.create_data_batch()
                    z = Variable(self.create_noise_batch(current_batch_size)).view(-1, G_inputs, 1, 1)

                    if cuda:
                        z = z.cuda()
                    generated_batch = self.G(z).detach()

                    real_prediction = self.D(x).squeeze() # 1x1

                    loss_d_r = self.D.loss(real_prediction, y_almost_ones)
                    loss_d_r.backward()

                    generated_prediction = self.D(generated_batch.view(current_batch_size, 1, 28, 28)).squeeze() # 1x1
                    loss_d_f = self.D.loss(generated_prediction, y_almost_zeros)

                    temp.append(loss_d_r.data + loss_d_f.data)
                    loss_d_f.backward()

                    self.D_optimiser.step()

            d_loss.append(np.mean(temp))
            temp = []

            for k in range(0, G_steps):
                z = Variable(self.create_noise_batch(current_batch_size)).view(-1, G_inputs, 1, 1)
                if cuda:
                    z = z.cuda()
                generated_batch = self.G(z)

                D_prediction = self.D(generated_batch.view(current_batch_size, 1, 28, 28)).squeeze() # 1x1
                loss_G = self.G.loss(D_prediction, y_almost_ones)
                temp.append(loss_G.data)
                loss_G.backward()

                predictions.append(D_prediction.mean().data)

                self.G_optimiser.step()
            g_loss.append(np.mean(temp))

    def create_noise_batch(self, batch_size):
        G_in = np.random.normal(0.0, 1.0, [batch_size, G_inputs])
        return torch.from_numpy(G_in).type(torch.FloatTensor)

root = './dataMnist'
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=minibatch_size,
                shuffle=True)

T = Trainer()
T.train(train_loader)

torch.save(T.G.state_dict(), "g_saved.pt")
torch.save(T.D.state_dict(), "d_saved.pt")

plt.plot(predictions, label="test")
plt.savefig("predictions.png")
plt.show()

plt.plot(d_loss, label="d_loss")
plt.plot(g_loss, label="g_loss")
plt.legend(loc="best")
plt.savefig("loss.png")
plt.show()
