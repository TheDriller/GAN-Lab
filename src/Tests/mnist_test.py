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

        self.D_optimiser = optim.Adam(self.D.parameters(), lr = lr)
        self.G_optimiser = optim.Adam(self.G.parameters(), lr = lr)
        self.predictions = []

    def train(self, train_loader):
        global index_list
        for i in range(0, epoch):
            temp = []
            for k in range(0, D_steps):
                print("Epoch: " + str(i) + ", D step: " + str(k))
#                for batch_id in range(int((train_size - 1) / minibatch_size) + 1):


                for batch_idx, (x, target) in enumerate(train_loader):
                    if(batch_idx > 10):
                        break
                    #print(batch_idx)
                    self.D.zero_grad()

#                    size, data_batch = self.create_data_batch()
                    z = Variable(self.create_noise_batch()).view(-1, G_inputs, 1, 1)

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
                    temp.append(loss_d_r.data + loss_d_f.data)
                    loss_d_f.backward()

                    self.D_optimiser.step()
                index_list = np.arange(0, train_size)
            d_loss.append(np.mean(temp))
            temp = []
            for k in range(0, G_steps):
                z = Variable(self.create_noise_batch()).view(-1, G_inputs, 1, 1)
                if cuda:
                    z = z.cuda()
                generated_batch = self.G(z)

                y_ones = Variable(torch.ones(minibatch_size))
                if cuda:
                    y_ones = y_ones.cuda()

                D_prediction = self.D(generated_batch.view(minibatch_size, 1, 28, 28)).squeeze() # 1x1
                loss_G = self.G.loss(D_prediction, y_ones)
                temp.append(loss_G.data)
                loss_G.backward()

                predictions.append(D_prediction.mean().data)

                self.G_optimiser.step()
            g_loss.append(np.mean(temp))

    def create_noise_batch(self):
        G_in = np.random.normal(0.0, 1.0, [minibatch_size, G_inputs])
        return torch.from_numpy(G_in).type(torch.FloatTensor)

root = './dataMnist'
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
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
