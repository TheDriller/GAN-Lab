
from hyperparameters import *
from models import *
import matplotlib.image as image
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import os

G = G_conv()
G.load_state_dict(torch.load('results/g_saved.pt'))

step = 0.1

for i in range(10):
    os.makedirs("results/latent_generated" + str(i) + "/", exist_ok = True)
    start_point = torch.rand(G_inputs).type(torch.FloatTensor)

    direction = np.random.randint(-1, 2, size=G_inputs)

    direction = direction
    cmpt = 0
    for index in np.arange(0, 3, step):

        added = torch.from_numpy(direction * index).type(torch.FloatTensor)

        point = torch.add(start_point, added)

        imagetemp = G(Variable(point.view(1, G_inputs, 1, 1))).view(image_x, image_y)
        image.imsave("results/latent_generated" + str(i) + "/" + str(cmpt) + ".png", imagetemp.data)
        cmpt += 1



