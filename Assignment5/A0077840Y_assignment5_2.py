import math
import numpy
import pylab as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.io.wavfile import write

import os
os.getcwd()                                             # get the directory of current process

output_dir_one = "ipython/part2a.wav"
output_dir_two = "ipython/part2b.wav"

FS = 32000.0
PITCH_DUR = 0.25

midis = [60, 62, 64, 65, 67, 69, 71, 72, 72, 0, 67, 0, 64, 0, 60]

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

def pitch_tone_one(note, duration, num_harmonics):
    freq = pitch_freq(note)
    ratio = 1 - pow(0.5, num_harmonics)
    wave = numpy.zeros(int(duration * FS))
    for index in range(1, num_harmonics + 1):
        wave = wave + sine_wave(freq * index, duration, 1.0 / pow(2, index), 0.0)
    return wave / ratio

def pitch_tone_two(note, duration, num_harmonics):
    freq = pitch_freq(note)
    ratio = 1 - pow(0.5, num_harmonics)
    wave = numpy.zeros(int(duration * FS))
    for index in range(1, num_harmonics + 1):
        wave = wave + sine_wave(freq * pow(2, index - 1), duration, 1.0 / pow(2, index), 0.0)
    return wave / ratio

def adsr_window(duration):
    peak = duration * FS * 0.1
    attack = numpy.arange(peak) / peak 
    decay = numpy.arange(peak, peak/2, -0.5) / peak
    sustain = numpy.empty(duration * FS * 0.6)
    sustain.fill(0.5)
    release = numpy.arange(peak/2, 0, -0.25) / peak
    return numpy.concatenate((attack, decay, sustain, release))

def show_spectrogram(data):
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

music_one = numpy.zeros(0)
music_two = numpy.zeros(0)

for midi in midis:
    music_one = numpy.concatenate((music_one, adsr * pitch_tone_one(midi, PITCH_DUR, 4)))
    music_two = numpy.concatenate((music_two, adsr * pitch_tone_two(midi, PITCH_DUR, 4)))

music_one = numpy.array(32767 * music_one, dtype = numpy.int16)
music_two = numpy.array(32767 * music_two, dtype = numpy.int16)

write(output_dir_one, FS, music_one)
write(output_dir_two, FS, music_two)

show_spectrogram(music_one)
show_spectrogram(music_two)

exit(1)
