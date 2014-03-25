import math
import numpy
import scipy.fftpack
import pylab as plt
from scipy.io.wavfile import read, write

import os
os.getcwd()                                              # get the directory of current process

input_dir = "original.wav"
txt_dir = "sines.txt"
output_dir = "reconstructed.wav"

def db_spectrum(data, window):
    fft = numpy.fft.fft(data * window)
    fft = fft[:len(fft) / 2 + 1]
    magfft = abs(fft) / (numpy.sum(window) / 2.0)       # with window dependent normalization
    epsilon = 1e-10
    db = 20*numpy.log10(magfft)
    return db

def reverse_db(db):
    amp = numpy.power(10, db / 20.)
    return amp

def show_spectrogram(data, title):
    #data = data / 32768.
    num_buffers = len(data) / 128                   # calculate the number of buffer slices
    buffers = numpy.zeros((65, num_buffers))           # create matrix to store buffer data

    for i in range(num_buffers):                         # put buffer slices in matrix
        start = int(i * 128)
        end = int(i * 128 + 128)
        buf = data[start:end]
        buffers[:,i] = db_spectrum(buf, numpy.hamming(128))

    plt.imshow(buffers, origin='lower', aspect='auto')
    plt.title(title)
    plt.xlabel("Time (buffer index)")
    plt.ylabel("Frequency (bin)")
    plt.show()
    return

sample_freq, original = read(input_dir)

data = original / 32768.

num_buffers = len(data) / 256 - 1
buffers = numpy.zeros((512, num_buffers))

for i in range(num_buffers):
    start = int(i * 256)
    end = start + 512
    buffers[:, i] = data[start: end]

buffers = buffers * 0.5
reconstructed = numpy.zeros(len(data))

for i in range(num_buffers):
    start = int(i * 256)
    end = start + 512
    reconstructed[start: end] += buffers[:, i]

reconstructed = numpy.array(reconstructed * 32768, dtype = numpy.int16)       # convert data into 16-bit integer

residual = original - reconstructed
residual = residual[256: len(residual) - 256]

residual_error = max(abs(residual))

print residual_error

 
