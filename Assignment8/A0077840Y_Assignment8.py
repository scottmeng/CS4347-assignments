import math
import numpy
import scipy.fftpack
import pylab as plt
from scipy.io.wavfile import read, write

import os
os.getcwd()                                              # get the directory of current process

input_dir = "clear_d1.wav"
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

sample_freq, data = read(input_dir)

data = data / 32768.
num_buffers = len(data) / 128
buffers = numpy.zeros((65, num_buffers))
analysis = numpy.zeros((2, num_buffers))
bins_to_freq = sample_freq / 128.

for i in range(num_buffers):
    start = int(i * 128)
    end = start + 128
    buffers[:, i] = db_spectrum(data[start:end], numpy.hamming(128))

analysis[0] = numpy.argmax(buffers, axis = 0) * bins_to_freq        # frequency array
analysis[1] = numpy.amax(buffers, axis = 0)                         # amplitude array

numpy.savetxt(txt_dir, numpy.transpose(analysis), fmt="%.1f", delimiter="\t")

construction_params = numpy.zeros((3, num_buffers * 128))           # parameters for reconstructing audio
phase_correction = numpy.zeros(num_buffers)

amplitudes = reverse_db(analysis[1])

for i in range(1, num_buffers):
    phase_correction[i] = analysis[0, i-1] * (i * 128 - 1) / sample_freq * 2 * numpy.pi + phase_correction[i-1]

for i in range(num_buffers):
    start = 128 * i
    end  = start + 128
    construction_params[0, start:end] = analysis[0, i]              # frequency parameters
    construction_params[1, start:end] = amplitudes[i]               # amplitude parameters
    construction_params[2, start:end] = phase_correction[i]         # phase correction parameters

t = numpy.arange(int(num_buffers * 128))                            # reconstructing audio using one single sine wave
constructed = construction_params[1] * numpy.sin(t * (construction_params[0] / sample_freq) * (2 * numpy.pi) + construction_params[2])

write(output_dir, sample_freq, constructed)

show_spectrogram(data, "original")
show_spectrogram(constructed, "constructed")

 
