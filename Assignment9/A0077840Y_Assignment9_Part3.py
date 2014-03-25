import math
import numpy
import scipy.fftpack
import pylab as plt
from scipy.io.wavfile import read, write

import os
os.getcwd()                                              # get the directory of current process

input_dir = "original.wav"

def time_to_freq(data, window):
    fft = numpy.fft.fft(data * window)
    return fft

def sine_window(length):
    t = numpy.arange(length)
    return numpy.sin((t + 0.5) / length * numpy.pi)

def freq_to_time(data, window):
    ifft = numpy.fft.ifft(data)
    return ifft * window

sample_freq, original = read(input_dir)

data = original / 32768.

num_buffers = len(data) / 256 - 1
reconstructed = numpy.zeros(len(data))

for i in range(num_buffers):
    start = int(i * 256)
    end = start + 512
    freq_buffer = time_to_freq(data[start: end], sine_window(512))
    reconstructed[start: end] += freq_to_time(freq_buffer, sine_window(512))

reconstructed = numpy.array(reconstructed * 32768, dtype = numpy.int16)       # convert data into 16-bit integer

residual = original - reconstructed
residual = residual[256: len(residual) - 256]

residual_error = max(abs(residual))

print residual_error

 
