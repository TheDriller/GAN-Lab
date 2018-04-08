import torch

# hyperparameters
D_STEPS = 2
TOT_EPOCHS = 2
G_STEPS = 2
SONG_LENGTH = 110224
MINIBATCH_SIZE = 32
# TRAIN_SET_SIZE = 1 # ???
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
HIDDEN_SIZE = 11
LATENT_DIMENSION = 10 # ???
SONG_PIECE_SIZE = 4
SAVE = 0
cuda = torch.cuda.is_available()
DROPOUT_PROB = 0.4 # ???
LSTM_LAYERS = 1 # 350?
USE_ZEROS = False
