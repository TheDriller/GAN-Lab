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

import torchvision.datasets as dset
import torchvision.transforms as transforms

nb_generate = 10
generator = basic_rnn_generator(HIDDEN_SIZE, SONG_PIECE_SIZE)
generator.load_state_dict(torch.load("g_saved.pt", map_location='cpu'))
generator.eval()

def create_noise_batch():
    G_in = np.random.normal(0.0, 1.0, [nb_generate, LATENT_DIMENSION])
    return torch.from_numpy(G_in).type(torch.FloatTensor)

def forward_G():
    z = Variable(create_noise_batch())

    generated_batch = torch.FloatTensor()

    # generate song with the generator RNN
    hidden = generator.initHidden(nb_generate)
    generated_batch = torch.zeros(nb_generate, SONG_LENGTH)

    generated_batch_tmp, hidden = generator.forward(z, hidden)
    generated_batch[:, 0:SONG_PIECE_SIZE] = generated_batch_tmp.data
    zeros = Variable(torch.zeros(nb_generate, LATENT_DIMENSION).type(torch.FloatTensor))
    for i in range(SONG_PIECE_SIZE, SONG_LENGTH, SONG_PIECE_SIZE):
        generated_batch_tmp, hidden = generator.forward(zeros, hidden)
        generated_batch[:, i:i+SONG_PIECE_SIZE] = generated_batch_tmp.data

    return generated_batch


noise_batch = create_noise_batch()

res = forward_G()

for i in range(0, nb_generate):
    np.save("saved" + str(i) + ".npy", res[i])
