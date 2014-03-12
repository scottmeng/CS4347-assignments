import math
import numpy
from scipy.fftpack import fft
from scipy.io.wavfile import read
import pylab as plt

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

def cal_mel_interval(fs):
    nyquist_freq = fs / 2.0;
    mel_max = 1127 * math.log((1 + nyquist_freq / 700), 10)
    return mel_max / 28

def get_center(mel_interval, index):
    mel_val = mel_interval * index
    return (pow(10, mel_val / 1127) - 1) * 700

def triangle_window(center, width):
    left = math.floor(center - width / 2)
    center = round(center)
    right = math.ceil(center + width / 2)
    print left, center, right
    ascending_half = numpy.delete(numpy.linspace(0, 1, num = (center - left + 1)), center - left)
    descending_half = numpy.linspace(1, 0, num = (right - center + 1))
    window = numpy.concatenate((numpy.zeros(left), ascending_half, descending_half))

    if len(window) < 513:
        return numpy.concatenate((window, numpy.zeros(512 - right)))
    else:
        return window[0:513]

input_file = open(input_dir)                             # read from ground truth file

file_names = input_file.readlines()

for file_name in file_names:
    music_dir = file_name.split()[0]                     # extract music and speech file directory
    label = file_name.split()[1]
    samp_rate, data = read(music_dir)

    mel_interval = 51.9
    bin_val = samp_rate / 1024.0
    print mel_interval
        
    data = data / 32768.                                 # normalize wav file data
    data = apply_pre_emphasis(data)

    num_buffers = len(data) / 512 - 1                    # calculate the number of buffer slices
        
    for i in range(num_buffers):                         # put buffer slices in matrix
        start = int(i * 512)
        end = int(i * 512 + 1024)
        buffer_data = data[start:end]

        buffer_data = db_spectrum(buffer_data, numpy.hamming(1024))
        print buffer_data.shape
        print buffer_data

        for i in range(1, 27):
            window = triangle_window(get_center(mel_interval, i) / bin_val, i * mel_interval / bin_val)
            print numpy.sum(window * buffer_data)
            exit(0)
        
