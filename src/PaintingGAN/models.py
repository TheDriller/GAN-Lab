import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.loss = nn.BCELoss()
        self.fc1 = nn.Linear(G_INPUTS, 8 * MODEL_COMPLEXITY * NB_CHANNELS)
        self.fc2 = nn.Linear(8 * MODEL_COMPLEXITY* NB_CHANNELS, 16 * MODEL_COMPLEXITY* NB_CHANNELS)
        self.fc3 = nn.Linear(16 * MODEL_COMPLEXITY* NB_CHANNELS, 32 * MODEL_COMPLEXITY* NB_CHANNELS)
        self.fc4 = nn.Linear(32 * MODEL_COMPLEXITY* NB_CHANNELS, IMAGE_Y * IMAGE_X* NB_CHANNELS)

    def forward(self, input):
        out = F.leaky_relu(self.fc1(input), LEAK)
        out = F.leaky_relu(self.fc2(out), LEAK)
        out = F.leaky_relu(self.fc3(out), LEAK)
        out = F.sigmoid(self.fc4(out))
        return out

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.loss = nn.BCELoss()
        self.fc1 = nn.Linear(IMAGE_Y * IMAGE_X* NB_CHANNELS, 32 * MODEL_COMPLEXITY* NB_CHANNELS)
        self.fc2 = nn.Linear(32 * MODEL_COMPLEXITY* NB_CHANNELS, 16 * MODEL_COMPLEXITY* NB_CHANNELS)
        self.fc3 = nn.Linear(16 * MODEL_COMPLEXITY* NB_CHANNELS, 8 * MODEL_COMPLEXITY* NB_CHANNELS)
        self.fc4 = nn.Linear(8 * MODEL_COMPLEXITY* NB_CHANNELS, 1)

    def forward(self, input):
        out = F.leaky_relu(self.fc1(input), LEAK)
        out = F.dropout(out, 0.3)
        out = F.leaky_relu(self.fc2(out), LEAK)
        out = F.dropout(out, 0.3)
        out = F.leaky_relu(self.fc3(out), LEAK)
        out = F.dropout(out, 0.3)
        out = F.sigmoid(self.fc4(out))
        return out

class G_conv(nn.Module):
    # initializers
    def __init__(self):
        super(G_conv, self).__init__()
        self.loss = nn.BCELoss()
        self.deconv1 = nn.ConvTranspose2d(G_INPUTS, 8 * NB_CHANNELS * MODEL_COMPLEXITY, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(8 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.deconv2 = nn.ConvTranspose2d(8 * NB_CHANNELS * MODEL_COMPLEXITY, 4 * NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(4 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.deconv3 = nn.ConvTranspose2d(4 * NB_CHANNELS * MODEL_COMPLEXITY, 2 * NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(2 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.deconv4 = nn.ConvTranspose2d(2 * NB_CHANNELS * MODEL_COMPLEXITY, 1 * NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(1 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.deconv5 = nn.ConvTranspose2d(1 * NB_CHANNELS * MODEL_COMPLEXITY, 1 * NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(1 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.deconv6 = nn.ConvTranspose2d(1 * NB_CHANNELS * MODEL_COMPLEXITY, 1 * NB_CHANNELS, 4, 2, 1)
        self.weights_init(MEAN, STD)

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.sigmoid(self.deconv6(x))

        return x

def weights_init_general(model, mean, std):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(mean, std)
            model._modules[m].bias.data.zero_()


class D_conv(nn.Module):
    # initializers
    def __init__(self):
        super(D_conv, self).__init__()
        self.loss = nn.BCELoss()
        self.conv1 = nn.Conv2d(NB_CHANNELS, NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.conv2 = nn.Conv2d(NB_CHANNELS * MODEL_COMPLEXITY, 2 * NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(2 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.conv22 = nn.Conv2d(2 * NB_CHANNELS * MODEL_COMPLEXITY, 2 * NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.conv22_bn = nn.BatchNorm2d(2 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.conv3 = nn.Conv2d(2 * NB_CHANNELS * MODEL_COMPLEXITY, 4 *NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(4 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.conv4 = nn.Conv2d(4 * NB_CHANNELS * MODEL_COMPLEXITY, 8 * NB_CHANNELS * MODEL_COMPLEXITY, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(8 * NB_CHANNELS * MODEL_COMPLEXITY)
        self.conv5 = nn.Conv2d(8 * NB_CHANNELS *  MODEL_COMPLEXITY, 1, 4, 1, 0)
        self.weights_init(MEAN, STD)

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv22_bn(self.conv22(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x
