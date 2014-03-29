import math
import numpy
import scipy.fftpack
import pylab as plt
from scipy.io.wavfile import read, write

import os
os.getcwd()                                              # get the directory of current process

lower_bound = 300
upper_bound = 1000

input_dir = "original.wav"
output_dir = "reconstructed_part3.wav"
residual_dir = "residual_part3.wav"

def time_to_freq(data, window, bins_to_freq):
    lower_index = int(lower_bound / bins_to_freq)
    upper_index = int(upper_bound / bins_to_freq)
    fft = numpy.fft.fft(data * window)
    fft[0: lower_index] = numpy.zeros(lower_index)
    fft[upper_index: 512 - upper_index] = numpy.zeros(512 - 2 * upper_index)
    fft[512 - lower_index: 512] = numpy.zeros(lower_index)
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
bins_to_freq = sample_freq / 512.

for i in range(num_buffers):
    start = int(i * 256)
    end = start + 512
    freq_buffer = time_to_freq(data[start: end], sine_window(512), bins_to_freq)
    reconstructed[start: end] += freq_to_time(freq_buffer, sine_window(512))

reconstructed = numpy.array(reconstructed * 32768, dtype = numpy.int16)       # convert data into 16-bit integer
residual = original - reconstructed

write(output_dir, sample_freq, reconstructed)
write(residual_dir, sample_freq, residual)

 
