import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.function as F
import torch.optim as optim
from torch.autograd import Variable

# hyperparameters
D_steps = 200
epoch = 5000
G_steps = 1
G_inputs = 1000 # ???
image_x = 332
image_y = 332
minibatch_size = 128
prob_real = 0.5
train_size = 1000 # ???
lr = 0.001

#global variable
index_list = np.arange(0, train_size)

#networks
class D(nn.module):
    def __init__(self):
        super(D, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, x):
        pass

class G(nn.module):
    def __init__(self):
        super(G, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, x):
        pass

class Trainer():
    def __init__(self):
        self.D = D()
        self.G = G()
        self.D_optimiser = optim.Adam(D.parameters(), lr = lr)
        self.G_optimiser = optim.Adam(G.parameters(), lr = lr)


    def train(self):
        for i in range(0, epoch):
            for k in range(0, D_steps):
                D.zero_grad()

                size, data_batch = create_data_batch()
                z = create_noise_batch()
                noise_batch = G(Variable(z))

                loss_d_r = self.D.loss(D(Variable(data_batch)))
                loss_d_r.backward()

                loss_d_f = self.D.loss(1 - D(noise_batch))
                loss_d_f.backward()
                self.D_optimiser.step()
                pass

    def create_noise_batch(self):
        G_in = np.random.normal(0.0, 1.0, [minibatch_size, G_inputs])
        return G_in

    def create_data_batch(self):
        size = np.min(minibatch_size, index_list.shape[0])

        if(size == 0):
            return 0, []

        indices = np.random.randint(0, index_list.shape[0], size = size)
        names = index_list[indices]
        index_list.remove(names)
        batch = np.zeros([size, image_x * image_y])

        for i in range (0, size):
            batch[i] = np.load(name[i] + ".npy")

        return size, batch

T = Trainer()
t.train()
