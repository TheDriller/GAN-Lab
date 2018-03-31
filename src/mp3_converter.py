import sys
import tempfile
import os
import pydub
import scipy
import scipy.io.wavfile
import numpy as np

########### CONSTANTS ###########
#################################

DATA_LENGTH = 332 * 332     # = 110224
STEP = 4

min_val = - (pow(2, 15) - 1)
max_val = pow(2, 15) - 1
norm_min_val = - (pow(2, 8) - 1)
norm_max_val = pow(2, 8) - 1

#################################

def read_mp3(filepath, as_float = False):
    """
    Read an MP3 File into numpy data.
    :param file_path: String path to a file
    :param as_float: Cast data to float and normalize to [-1, 1]
    :return: Tuple(rate, data), where
        rate is an integer indicating samples/s
        data is an ndarray(n_samples, 2)[int16] if as_float = False
            otherwise ndarray(n_samples, 2)[float] in range [-1, 1]
    """

    path, ext = os.path.splitext(filepath)
    assert ext=='.mp3'
    mp3 = pydub.AudioSegment.from_mp3("/Users/lucaalbinati/Documents/testingPython/palace.mp3")
    _, path = tempfile.mkstemp()
    mp3.export(path, format="wav")
    rate, data = scipy.io.wavfile.read(path)
    os.remove(path)
    if as_float:
        data = data/(2**15)
    return rate, data

def write_mp3(filepath, rate, data):
    scipy.io.wavfile.write(filepath, rate, data)

def compress_mp3(rate, data, step):
    assert step >= 1
    assert len(data) >= DATA_LENGTH
    assert len(data.shape) <= 2

    # Remove an extra channel, if present
    if len(data.shape) == 2:
        data = data_one_channel(data)

    # Resample data using step variable
    # Also trim data, if longer than DATA_LENGTH
    newRate = rate / step
    newData = data[::step]
    newData = newData[:DATA_LENGTH]

    return newRate, newData

def data_one_channel(data):
    # Remove a channel from data, which is previously a 2 dimensional
    # array (we believe it means left and right audio channel). Taking
    # either one seems to work fine
    return data[:,0]

def compress_data_mean(rate, data, step):
    newRate = rate / step
    newData = np.zeros([int(len(data) / step), 2])

    index = 0
    for i in range(0, len(data), step):
        mean = np.mean(data[i:i+step])
        newData[index] = [mean, mean]

    return newRate, newData

def normalize_mp3(data):
    interpolation = scipy.interpolate.interp1d([min_val, max_val], [norm_min_val, norm_max_val])
    return interpolation(data)

def denormalize_mp3(data):
    interpolation = scipy.interpolate.interp1d([norm_min_val, norm_max_val], [min_val, max_val])
    return interpolation(data)

def open_and_convert(mp3_filepath):
    rate, data = read_mp3(mp3_filepath)
    rate, data = compress_mp3(rate, data, STEP)
    data = normalize_mp3(data)
    return data

def convert_dataset(directory_filepath):
    # Iterate through all .mp3 files within the directory
    # and convert them into .npy files

    print("Starting dataset conversion")
    directory = os.fsencode(directory_filepath)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mp3"):
            print("Converting file: " + filename)
            filepath = directory_filepath + filename
            mp3_data = open_and_convert(filepath)
            os.mkdir(directory_filepath + "npy")
            output_filename = directory_filepath + "npy/" + filename.replace('.mp3', '')
            np.save(output_filename, mp3_data)

    print("Finished dataset conversion")

def main():
    # Takes as argument the directory_filepath
    if len(sys.argv) == 2:
        directory_filepath = sys.argv[1]

        if not directory_filepath.endswith('/'):
            directory_filepath = directory_filepath + "/"

        convert_dataset(directory_filepath)

        sys.exit(0)
    else:
        sys.exit(-1)

main()
