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
        self.d_loss = nn.BCELoss()

    def forward(self, x):
        pass

class G(nn.module):
    def __init__(self):
        super(G, self).__init__()
        self.g_loss = nn.BCELoss()

    def forward(self, x):
        pass


def train(batch):
    for i in range(0, epoch):
        for k in range(0, D_steps):
            D.zero_grad()

            size, data_batch_temp = create_data_batch()
            z = create_noise_batch()

            noise_batch = G(Variable(z))
            data_batch = Variable(data_batch_temp)
            pass


def create_noise_batch():
    G_in = np.random.normal(0.0, 1.0, [minibatch_size, G_inputs])
    return G_in

def create_data_batch():
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

D = D()
G = G()

D_optimiser = optim.adam(D.parameters(), lr=lr)

create_noise_batch()
