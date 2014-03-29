import math
import numpy
import scipy.fftpack
import pylab as plt
from scipy.io.wavfile import read, write

import os
os.getcwd()                                             # get the directory of current process

lower_bound = 200                                       # configuration parameters
upper_bound = 1200
quan_bits = 8

input_dir = "original.wav"
output_dir = "reconstructed_part4.wav"
residual_dir = "residual_part4.wav"

max_abs = 0;                                            # get the maximum absolute value of imaginary and real parts

def time_to_freq(data, window, bins_to_freq, quantization):
    global max_abs
    lower_index = int(lower_bound / bins_to_freq)       # compute index range from frequency range
    upper_index = int(upper_bound / bins_to_freq)
    fft = numpy.fft.fft(data * window)
    fft[0: lower_index] = numpy.zeros(lower_index)
    fft[upper_index: 512 - upper_index] = numpy.zeros(512 - 2 * upper_index)
    fft[512 - lower_index: 512] = numpy.zeros(lower_index)

    if quantization == False:                           # get maximum
        if max(abs(fft.imag)) > max_abs:
            max_abs = max(abs(fft.imag))
        if max(abs(fft.real)) > max_abs:
            max_abs = max(abs(fft.real))
    else:                                               # perform quantization
        real_parts = numpy.array(fft.real / max_abs * (2 ** (quan_bits - 1)), dtype = numpy.int32)
        fft.real = real_parts / float(2 ** (quan_bits - 1)) * max_abs
        imag_parts = numpy.array(fft.imag / max_abs * (2 ** (quan_bits - 1)), dtype = numpy.int32)
        fft.imag = imag_parts / float(2 ** (quan_bits - 1)) * max_abs
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
    start = i * 256
    end = start + 512
    freq_buffer = time_to_freq(data[start: end], sine_window(512), bins_to_freq, False)

for i in range(num_buffers):
    start = i * 256
    end = start + 512
    freq_buffer = time_to_freq(data[start: end], sine_window(512), bins_to_freq, True)
    reconstructed[start: end] += freq_to_time(freq_buffer, sine_window(512))

reconstructed = numpy.array(reconstructed * 32768, dtype = numpy.int16)       # convert data into 16-bit integer
residual = original - reconstructed

write(output_dir, sample_freq, reconstructed)
write(residual_dir, sample_freq, residual)
 
