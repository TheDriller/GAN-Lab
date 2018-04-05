import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# hyperparameters
D_steps = 10
epoch = 10
G_steps = 1
G_inputs = 1000 # ???
image_x = 332
image_y = 332
minibatch_size = 1
train_size = 1 # ???
lr = 0.001

#global variable
index_list = np.arange(0, train_size)

#estimation
predictions = []

#networks
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.loss = nn.BCELoss()
        self.fc1 = nn.Linear(image_x * image_y, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim = 0)
        return out

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.loss = nn.BCELoss()
        self.fc1 = nn.Linear(G_inputs, image_y * image_x)

    def forward(self, x):
        return F.relu(self.fc1(x))

class Trainer():
    def __init__(self):
        self.D = D()
        self.G = G()
        self.D_optimiser = optim.Adam(self.D.parameters(), lr = lr)
        self.G_optimiser = optim.Adam(self.G.parameters(), lr = lr)
        self.predictions = []

    def train(self):
        global index_list
        for i in range(0, epoch):
            for k in range(0, D_steps):
                print(k)
                for batch_id in range(int((train_size - 1) / minibatch_size) + 1):
                    self.D.zero_grad()

                    size, data_batch = self.create_data_batch()
                    z = self.create_noise_batch()
                    generated_batch = self.G(Variable(z)).detach()
                    real_prediction = self.D(Variable(data_batch))[0] # 1x1

                    loss_d_r = self.D.loss(real_prediction, Variable(torch.ones(1)))
                    loss_d_r.backward()

                    generated_prediction = self.D(generated_batch)[0] # 1x1
                    loss_d_f = self.D.loss(1 - generated_prediction, Variable(torch.ones(1)))
                    loss_d_f.backward()

                    self.D_optimiser.step()
                index_list = np.arange(0, train_size)

            for k in range(0, G_steps):
                z = self.create_noise_batch()
                generated_batch = self.G(Variable(z))

                D_prediction = self.D(generated_batch)[0] # 1x1
                loss_G = self.G.loss(D_prediction, Variable(torch.ones(1)))
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
            batch[i] = np.load("../../data/" + str(names[i]) + ".npy")

        return size, torch.from_numpy(batch).type(torch.FloatTensor)

T = Trainer()
T.train()

plt.plot(predictions, label="test")
plt.show()
