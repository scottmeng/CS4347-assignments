import math
import numpy
from scipy.fftpack import fft
from scipy.io.wavfile import read

import os
os.getcwd()                                              # get the directory of current process

input_dir = "music_speech.mf"

def apply_pre_emphasis(data):
    copy = numpy.delete(data, len(data) - 1)
    copy = 0.95 * numpy.insert(copy, [0.0], 0)
    return data - copy

def db_spectrum(data, window):
    fft = numpy.fft.fft(data * window)
    fft = fft[:len(fft) / 2 + 1]
    magfft = abs(fft) / (numpy.sum(window) / 2.0)
    epsilon = pow(10, -10)
    db = 20*numpy.log10(magfft + epsilon)
    return db

input_file = open(input_dir)                             # read from ground truth file

file_names = input_file.readlines()

for file_name in file_names:
    music_dir = file_name.split()[0]                     # extract music and speech file directory
    label = file_name.split()[1]
    samp_rate, data = read(music_dir)

    print samp_rate
        
    data = data / 32768.                                 # normalize wav file data
    data = apply_pre_emphasis(data)

    num_buffers = len(data) / 512 - 1                    # calculate the number of buffer slices
        
    for i in range(num_buffers):                         # put buffer slices in matrix
        start = int(i * 512)
        end = int(i * 512 + 1024)
        buffer_data = data[start:end]

        buffer_data = db_spectrum(buffer_data, numpy.hamming(1024))
        print buffer_data.shape
        exit(0)
