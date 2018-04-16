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
import os
import glob

import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch.utils.data as datautils

predictions = []
g_loss = []
d_loss = []
last_d_loss = 0
last_g_loss = 0

# load songs previously transformed into .npy form
def load_real_songs():
    directory_str = "data/npy/"
    files = glob.glob(directory_str + "*.npy")
    data = np.ndarray((len(files), SONG_LENGTH))

    for i, file in enumerate(files):
        data[i] = np.load(file)

    dataset = datautils.TensorDataset(torch.from_numpy(data).type(torch.FloatTensor), torch.ones(data.shape))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True, num_workers=2)
    return train_loader


# create targets for the losses
def get_targets(length):
    y_almost_ones = Variable(torch.from_numpy(np.random.uniform(0.8, 1.0, length)).type(torch.FloatTensor))
    if cuda:
        y_almost_ones = y_almost_ones.cuda()

    y_almost_zeros = Variable(torch.from_numpy(np.random.uniform(0.0, 0.2, length)).type(torch.FloatTensor))
    if cuda:
        y_almost_zeros = y_almost_zeros.cuda()
    return y_almost_ones, y_almost_zeros

#Training
class Trainer():
    def __init__(self):
        # D and G with their optimiser
        self.D = LSTM_discriminator(HIDDEN_SIZE)
        if cuda:
            self.D.cuda()
        self.G = LSTM_generator(HIDDEN_SIZE, SONG_PIECE_SIZE)
        if cuda:
            self.G.cuda()

        self.D_optimiser = optim.Adam(self.D.parameters(), lr = LR, betas = (BETA1, BETA2))
        self.G_optimiser = optim.Adam(self.G.parameters(), lr = LR, betas = (BETA1, BETA2))


    def train_generator(self,batch):
        current_batch_size = batch.shape[1]
        losses = []
        y_almost_ones,y_almost_zeros = get_targets(current_batch_size)

        # train generator for G_Steps
        for k in range(0, G_STEPS):
            print("Training generator - k = "+str(k))
            self.G.zero_grad()

            z = Variable(self.create_noise_batch(current_batch_size)).view(current_batch_size, 1, -1)
            if cuda:
                z = z.cuda()

            generated_batch = Variable(self.G.forward_G(current_batch_size, z))

            if cuda:
                generated_batch = generated_batch.cuda()

            # lstm input is (seq_index, batch_index, (network)input_index)
            generated_batch = generated_batch.view(int(generated_batch.size(1) / SONG_PIECE_SIZE), generated_batch.size(0), -1)

            generated_prediction = self.D.forward_D(generated_batch, USE_FEATURE_MATCHING)

            if USE_FEATURE_MATCHING:
                feature_real = self.D.forward_D(batch.detach(), USE_FEATURE_MATCHING)
                feature_real = torch.mean(feature_real, 0)
                feature_fake = torch.mean(generated_prediction, 0)
                loss_G = self.G.loss(feature_fake, feature_real.detach())
            else:
                loss_G = self.G.loss(generated_prediction.squeeze(), y_almost_ones)

            loss_G.backward()

            predictions.append(generated_prediction.mean().data[0])
            last_g_loss = loss_G.data[0]

            losses.append(last_g_loss)
            if last_g_loss < 0.7 * last_d_loss:
                self.G_optimiser.step()
        return losses

    def train_discriminator(self, batch):
        losses = []
        y_almost_ones,y_almost_zeros = get_targets(batch.shape[1])
        current_batch_size = batch.shape[1]

        #train discriminator for D_STEPS
        for k in range(0, D_STEPS):
            print("Training discriminator - k = "+str(k))
            self.D.zero_grad()
            real_prediction = self.D.forward_D(batch, False)

            z = Variable(self.create_noise_batch(current_batch_size)).view(current_batch_size, 1, -1)
            if cuda:
                z = z.cuda()

            generated_batch = Variable(self.G.forward_G(current_batch_size, z))

            if cuda:
                generated_batch = generated_batch.cuda()
            loss_d_r = self.D.loss(real_prediction.squeeze(), y_almost_ones)

            # lstm input is (seq_index, batch_index, (network)input_index)
            generated_batch = generated_batch.view(int(generated_batch.size(1) / SONG_PIECE_SIZE), generated_batch.size(0), -1)
            generated_prediction = self.D.forward_D(generated_batch, False)

            loss_d_f = self.D.loss(generated_prediction.squeeze(), y_almost_zeros)
            loss_total = loss_d_r + loss_d_f
            loss_total.backward()

            losses.append(loss_d_r.data + loss_d_f.data)
            last_d_loss = (loss_d_r + loss_d_f).data[0]

            if last_d_loss > 0.7 * last_g_loss:
                self.D_optimiser.step()
        return losses

    def train(self,real_songs):
        z_saved = Variable(self.create_noise_batch(5).view(5, 1, LATENT_DIMENSION))
        if cuda:
            z_saved = z_saved.cuda()
        last_d_loss = 0
        last_g_loss = 0

        for i in range(0, TOT_EPOCHS):
            print("TOT_EPOCHS: " + str(i))

            for batch_id, (x, target) in enumerate(real_songs):
                print("Starting batch "+str(batch_id))

                batch = x
                batch = batch.view(int(batch.shape[1] / SONG_PIECE_SIZE), batch.shape[0], -1)
                batch = Variable(batch.float())
                if cuda:
                    batch = batch.cuda()

                current_batch_size = batch.shape[0]

                # Train discriminator
                discriminator_losses = self.train_discriminator(batch)
                d_loss_t = np.mean(discriminator_losses)
                d_loss.append(d_loss_t)

                # Train generator
                generator_losses = self.train_generator(batch)

                g_loss_t = np.mean(generator_losses)
                g_loss.append(g_loss_t)

                self.write_log(d_loss, g_loss)

            self.write_epoch_song(z_saved, i)

    def create_noise_batch(self, batch_size):
        G_in = np.random.normal(0.0, 1.0, [batch_size, LATENT_DIMENSION])
        return torch.from_numpy(G_in).type(torch.FloatTensor)

    def write_epoch_song(self, z, epoch):
        dir_saved = "data/saved_epochs/"
        os.makedirs(dir_saved, exist_ok = True)
        generated_songs = self.G.forward_G(z.size(0), z)
        for i in range(0, generated_songs.size(0)):
            np.save(dir_saved + "saved_song" + str(epoch) + str(i) + ".npy", generated_songs[i])

    def write_log(self, d_loss, g_loss):
        dir_saved = "data/log/"
        os.makedirs(dir_saved, exist_ok = True)

        D_log = "D_loss.log"
        G_log = "G_loss.log"

        if os.path.exists(dir_saved + D_log):
            opt = 'a' # append if already exists
        else:
            opt = 'w' # make a new file if not

        with open(dir_saved + "D_loss.log", opt) as log:
            log.write(str(d_loss[-1]) + "\n")

        if os.path.exists(dir_saved + G_log):
            opt = 'a' # append if already exists
        else:
            opt = 'w' # make a new file if not

        with open(dir_saved + "G_loss.log", opt) as log:
            log.write(str(g_loss[-1]) + "\n")

if __name__ == '__main__':
    T = Trainer()
    real_songs = load_real_songs()
    T.train(real_songs)

    torch.save(T.G.state_dict(), "g_saved.pt")
    torch.save(T.D.state_dict(), "d_saved.pt")

    print("done")

    if SAVE:
        plt.plot(predictions, label="test")
        plt.savefig("predictions.png")
        plt.show()

        plt.plot(d_loss, label="d_loss")
        plt.plot(g_loss, label="g_loss")
        plt.legend(loc="best")
        plt.savefig("loss.png")
        plt.show()
