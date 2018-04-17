import torch

# hyperparameters
D_STEPS = 5
TOT_EPOCHS = 5
G_STEPS = 5
SONG_LENGTH = 110224 # 2^4 * 83^2
MINIBATCH_SIZE = 32
# TRAIN_SET_SIZE = 1 # ???
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
HIDDEN_SIZE = 10
LATENT_DIMENSION = 10 # ???
SONG_PIECE_SIZE = 83 * 4
SAVE = 1
cuda = torch.cuda.is_available()
DROPOUT_PROB = 0.4
LSTM_LAYERS = 1
USE_ZEROS = False
USE_FEATURE_MATCHING = True
FM_DIV = 2

SAVE = True
CHECKPOINT = 3

SAVE_PATH = "results/"

LOAD_MODELS = True

D_LOAD_NAME = "epoch6_D_model.pt"
G_LOAD_NAME = "epoch6_G_model.pt"
