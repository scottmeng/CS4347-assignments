import math
import numpy
import pylab as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.io.wavfile import write

import os
os.getcwd()                                             # get the directory of current process

output_dir = "ipython/dialling.wav"
FS = 8000.0
NUM_DUR = 0.25
DIAL_DUR = 0.5

number_tones = {
    0 : [961, 1336],
    1 : [697, 1209],
    2 : [697, 1336],
    3 : [697, 1477],
    4 : [770, 1209],
    5 : [770, 1336],
    6 : [770, 1477],
    7 : [852, 1209],
    8 : [852, 1336],
    9 : [852, 1477]
}

numbers = [6, 5, 1, 2, 3, 4, 7, 8, 9, 0]

def sine_wave(freq, duration = 1.0, amplitude = 1.0, phase = 0.0):
    t = numpy.arange(int(duration * FS))
    return amplitude * numpy.sin(t * (freq / FS) * (2 * numpy.pi) + phase)

def dual_tone(freq1, freq2, duration):
    return sine_wave(freq1, duration, 0.5, 0.0) + sine_wave(freq2, duration, 0.5, 0.0)

def dial_tone(duration):
    return sine_wave(350, duration, 0.5, 0.0) + sine_wave(440, duration, 0.5, 0.0)

def db_spectrum(data, window):
    fft = numpy.fft.fft(data * window)
    fft = fft[:len(fft) / 2 + 1]
    magfft = abs(fft) / (numpy.sum(window) / 2.0)
    epsilon = pow(10, -10)
    db = 20*numpy.log10(magfft + epsilon)
    return db

def show_spectrogram(data):
    data = data / 32768.
    num_buffers = len(data) / 256 - 1                   # calculate the number of buffer slices
    buffers = numpy.zeros((num_buffers, 257))           # create matrix to store buffer data

    for i in range(num_buffers):                        # put buffer slices in matrix
        start = int(i * 256)
        end = int(i * 256 + 512)
        buf = data[start:end]
        buffers[i] = db_spectrum(buf, numpy.blackman(512))

    buffers = buffers.transpose()
    im = plt.imshow(buffers, origin='lower')
    plt.show()
    return

data = dial_tone(DIAL_DUR)

for number in numbers:
    temp = dual_tone(number_tones[number][0], number_tones[number][1], NUM_DUR)
    data = numpy.concatenate((data, temp))

data = numpy.array(data * 32767, dtype = numpy.int16)       # convert data into 16-bit integer

write(output_dir, FS, data)

show_spectrogram(data)

exit(0)
