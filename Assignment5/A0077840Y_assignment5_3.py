import math
import numpy
import pylab as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.io.wavfile import write

import os
os.getcwd()                                             # get the directory of current process

FS = 8000.0
PITCH_DUR = 0.25
midis = [60, 62, 64, 65, 67, 69, 71, 72, 72, 0, 67, 0, 64, 0, 60]

carrier_freq_factor   = [1.0,  1.5,  1.0,   1.0]
modulator_freq_factor = [0.5,  0.5,  2.0, 0.345]
carrier_amp_factor    = [1.0,  1.0,  1.0,   1.0]
modulator_amp_factor  = [0.5, 0.25, 0.25,   0.5]

output_dir = ["ipython/part3_a.wav", "ipython/part3_b.wav", "ipython/part3_c.wav", "ipython/part3_d.wav"]

def sine_wave(freq, duration = 1.0, amplitude = 1.0, phase = 0.0):
    t = numpy.arange(int(duration * FS))
    return amplitude * numpy.sin(t * (freq / FS) * (2 * numpy.pi) + phase)

def db_spectrum(data, window):
    fft = numpy.fft.fft(data * window)
    fft = fft[:len(fft) / 2 + 1]
    magfft = abs(fft) / (numpy.sum(window) / 2.0)
    epsilon = pow(10, -10)
    db = 20*numpy.log10(magfft + epsilon)
    return db

def pitch_freq(note):
    if note == 0:
        return 0.0
    return 440 * pow(2, ((note - 69) / 12.0))

def pitch_tone(note, duration, version):
    freq = pitch_freq(note)
    freq_car = freq * carrier_freq_factor[version]
    freq_mod = freq * modulator_freq_factor[version]
    amp_car = carrier_amp_factor[version]
    amp_mod = modulator_amp_factor[version]
    return (sine_wave(freq_car, duration, amp_car, 0.0) + ((amp_mod + amp_car) / 2) * (sine_wave((freq_car - freq_mod), duration, 1.0, 0.0) + sine_wave((freq_car + freq_mod), duration, 1.0, 0.0))) / (1 + amp_car + amp_mod)

def adsr_window(duration):
    peak = duration * FS * 0.1
    attack = numpy.arange(peak) / peak 
    decay = numpy.arange(peak, peak/2, -0.5) / peak
    sustain = numpy.empty(duration * FS * 0.6)
    sustain.fill(0.5)
    release = numpy.arange(peak/2, 0, -0.25) / peak
    return numpy.concatenate((attack, decay, sustain, release))

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

adsr = adsr_window(0.25)

for version_index in range(0, 4):
    music = numpy.zeros(0)
    for midi in midis:
        music = numpy.concatenate((music, adsr * pitch_tone(midi, PITCH_DUR, version_index)))
    music = numpy.array(32768 * music, dtype = numpy.int16)

    write(output_dir[version_index], FS, music)

    show_spectrogram(music)    

exit(1)

