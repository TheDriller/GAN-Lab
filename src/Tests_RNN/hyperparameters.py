import torch

# hyperparameters
D_STEPS = 10
TOT_EPOCHS = 1
G_STEPS = 5
SONG_LENGTH = 110224
MINIBATCH_SIZE = 128
# TRAIN_SET_SIZE = 1 # ???
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
HIDDEN_SIZE = 11
LATENT_DIMENSION = 10 # ???
SONG_PIECE_SIZE = 8
SAVE = 1
cuda = torch.cuda.is_available()
