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

def create_noise_batch():
    G_in = np.random.normal(0.0, 1.0, [nb_generate, G_inputs])
    return torch.from_numpy(G_in).type(torch.FloatTensor)

generator = G_conv()
generator.load_state_dict(torch.load("g_saved.pt")) #CHANGE NAME
generator.eval()

noise_batch = create_noise_batch()

res = generator(Variable(noise_batch.view(nb_generate, G_inputs, 1, 1)))
res = res.view(nb_generate, image_x, image_y)

for i in range(0, nb_generate):
    image.imsave("image_res/" + str(i) + ".png", res[i].data)
