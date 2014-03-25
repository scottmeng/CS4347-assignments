import math
import numpy
import scipy.fftpack
import pylab as plt
from scipy.io.wavfile import read, write

import os
os.getcwd()                                              # get the directory of current process

input_dir = "original.wav"

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

 
