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
RATE = int(44100 / STEP)

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
    mp3 = pydub.AudioSegment.from_mp3(filepath)
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
    assert len(data) * (3 / 5) >= DATA_LENGTH
    assert len(data.shape) <= 2

    # Remove an extra channel, if present
    if len(data.shape) == 2:
        data = data_one_channel(data)

    # Resample data using step variable
    # Also trim data, if longer than DATA_LENGTH
    newRate = rate / step
    newData = data[::step]

    # Extract middle of song, to avoid getting a "slow" start
    start = int(len(newData) * (2 / 5))
    end = start + DATA_LENGTH

    newData = newData[start : end]

    return newRate, newData

def data_one_channel(data):
    # Remove a channel from data, which is previously a 2 dimensional
    # array (we believe it means left and right audio channel). Taking
    # either one seems to work fine
    return data[:,0]

def compress_data_mean(rate, data, step):
    # Unused, didn't give good results

    newRate = rate / step
    newData = np.zeros([int(len(data) / step), 2])

    index = 0
    for i in range(0, len(data), step):
        mean = np.mean(data[i:i+step])
        newData[index] = [mean, mean]

    return newRate, newData

def normalize_mp3(data):
    # Get rid of -2^15 value, so that 0 becomes 0 during interpolation
    # (Problem linked with binary representation of integer, negative and positive...)
    for i in range(0, len(data)):
        if data[i] == - pow(2, 15):
            data[i] = - pow(2, 15) + 1

    interpolation = scipy.interpolate.interp1d([min_val, max_val], [norm_min_val, norm_max_val])
    return interpolation(data)

def denormalize_mp3(data):
    interpolation = scipy.interpolate.interp1d([norm_min_val, norm_max_val], [min_val, max_val])
    return interpolation(data)

def read_and_convert(mp3_filepath):
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
            mp3_data = read_and_convert(filepath)
            os.makedirs(directory_filepath + "npy", exist_ok = True)
            output_filename = directory_filepath + "npy/" + filename.replace('.mp3', '')
            np.save(output_filename, mp3_data)

    print("Finished dataset conversion")

def generate_mp3(npy_filepath):
    rate = RATE
    data = np.load(npy_filepath)
    data = denormalize_mp3(data)
    data = data.astype(np.int16)

    os.makedirs("generated_mp3s", exist_ok = True)
    output_filename = "generated_mp3s/" + npy_filepath.replace('npy/', '').replace('.npy', '.mp3')
    write_mp3(output_filename, rate, data)

def main():
    # If first argument is "-c" then convert an entire directory (second argument) of .mp3 into .npy
    # If first argument is "-g" then generate an .mp3 from a .npy (second argument)
    # Else fail

    if len(sys.argv) == 3:
        option = sys.argv[1]

        if option == "-c":
            directory_filepath = sys.argv[2]
            if not directory_filepath.endswith('/'):
                directory_filepath = directory_filepath + "/"
            convert_dataset(directory_filepath)
        elif option == "-g":
            assert(sys.argv[2].endswith(".npy"))
            generate_mp3(sys.argv[2])
        else:
            print("Wrong arguments")

        sys.exit(0)
    else:
        print("Wrong number of arguments")
        sys.exit(-1)

main()
