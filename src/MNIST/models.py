import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.loss = nn.BCELoss()
        self.fc1 = nn.Linear(G_inputs, 8 * model_complexity)
        self.fc2 = nn.Linear(8 * model_complexity, 16 * model_complexity)
        self.fc3 = nn.Linear(16 * model_complexity, 32 * model_complexity)
        self.fc4 = nn.Linear(32 * model_complexity, image_y * image_x)

    def forward(self, input):
        out = F.leaky_relu(self.fc1(input), leak)
        out = F.leaky_relu(self.fc2(out), leak)
        out = F.leaky_relu(self.fc3(out), leak)
        out = F.tanh(self.fc4(out))
        return out

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.loss = nn.BCELoss()
        self.fc1 = nn.Linear(image_y * image_x, 32 * model_complexity)
        self.fc2 = nn.Linear(32 * model_complexity, 16 * model_complexity)
        self.fc3 = nn.Linear(16 * model_complexity, 8 * model_complexity)
        self.fc4 = nn.Linear(8 * model_complexity, 1)

    def forward(self, input):
        out = F.leaky_relu(self.fc1(input), leak)
        out = F.dropout(out, 0.3)
        out = F.leaky_relu(self.fc2(out), leak)
        out = F.dropout(out, 0.3)
        out = F.leaky_relu(self.fc3(out), leak)
        out = F.dropout(out, 0.3)
        out = F.sigmoid(self.fc4(out))
        return out

class D_conv2(nn.Module):
    def __init__(self):
        super(D_conv2, self).__init__()
        self.loss = nn.BCELoss()

        self.conv1 = nn.Conv2d(1, 2 * model_complexity, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(2 * model_complexity)
        self.layer1 = nn.Sequential(self.conv1, self.bn1)

        self.conv2 = nn.Conv2d(2 * model_complexity, 4 * model_complexity, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4 * model_complexity)
        self.layer2 = nn.Sequential(self.conv2, self.bn2)

        self.conv3 = nn.Conv2d(4 * model_complexity, 8 * model_complexity, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(8 * model_complexity)
        self.layer3 = nn.Sequential(self.conv3, self.bn3)
        
        self.conv4 = nn.Conv2d(8 * model_complexity, 16 * model_complexity, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16 * model_complexity)
        self.layer4 = nn.Sequential(self.conv4, self.bn4)

        self.fully = nn.Linear(16 * image_x * image_y * model_complexity, 1)
        self.weights_init(mean, std)

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

    def forward(self, x):
        out = F.leaky_relu(self.layer1(x), leak)
        out = F.leaky_relu(self.layer2(out), leak)
        out = F.leaky_relu(self.layer3(out), leak)
        out = F.leaky_relu(self.layer4(out), leak)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(self.fully(out))
        return out

class G_conv2(nn.Module):
    def __init__(self):
        super(G_conv2, self).__init__()
        self.loss = nn.BCELoss()

        self.conv1 = nn.ConvTranspose2d(G_inputs, 8 * model_complexity, 5, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(8 * model_complexity)
        self.layer1 = nn.Sequential(self.conv1, self.bn1)

        self.conv2 = nn.ConvTranspose2d(8 * model_complexity, 4 * model_complexity, 5, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(4 * model_complexity)
        self.layer2 = nn.Sequential(self.conv2, self.bn2)

        self.conv3 = nn.ConvTranspose2d(4 * model_complexity, 2 * model_complexity, 4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(2 * model_complexity)
        self.layer3 = nn.Sequential(self.conv3, self.bn3)

        self.conv4 = nn.ConvTranspose2d(2 * model_complexity, 1 * model_complexity, 3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(1 * model_complexity)
        self.layer4 = nn.Sequential(self.conv4, self.bn4)

        self.deconv5 = nn.ConvTranspose2d(1 * model_complexity, 1, 4, padding=1, stride=2)
        self.weights_init(mean, std)

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = F.tanh(self.deconv5(out))
        return out

    
class G_conv(nn.Module):
    # initializers
    def __init__(self):
        super(G_conv, self).__init__()
        self.loss = nn.BCELoss()
        self.deconv1 = nn.ConvTranspose2d(100, 8 * model_complexity, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(8 * model_complexity)
        self.deconv2 = nn.ConvTranspose2d(8 * model_complexity, 4 * model_complexity, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(4 * model_complexity)
        self.deconv3 = nn.ConvTranspose2d(4 * model_complexity, 2 * model_complexity, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(2 * model_complexity)
        self.deconv4 = nn.ConvTranspose2d(2 * model_complexity, 1 * model_complexity, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(1 * model_complexity)
        self.deconv5 = nn.ConvTranspose2d(1 * model_complexity, 1, 4, 2, 1)
        self.weights_init(mean, std)

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

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
        self.conv1 = nn.Conv2d(1, 1 * model_complexity, 4, 2, 1)
        self.conv2 = nn.Conv2d(1 * model_complexity, 2 * model_complexity, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(2 * model_complexity)
        self.conv3 = nn.Conv2d(2 * model_complexity, 4 * model_complexity, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(4 * model_complexity)
        self.conv4 = nn.Conv2d(4 * model_complexity, 8 * model_complexity, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(8 * model_complexity)
        self.conv5 = nn.Conv2d(8 * model_complexity, 1, 4, 1, 0)
        self.weights_init(mean, std)

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x