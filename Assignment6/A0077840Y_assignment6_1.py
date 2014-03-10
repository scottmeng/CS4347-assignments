import math
import numpy
from scipy import signal
import pylab as plt
from scipy.io.wavfile import write

import os
os.getcwd()

FS = 44100.0
M = 10
AMP = 0.5
f = 1000.0

output_dir_reconstructed = "part1_reconstructed.wav"
output_dir_perfect = "part1_perfect.wav"

def sawtooth(freq, M, duration = 1.0, amplitude = 0.5):
    wave = numpy.zeros(int(duration * FS))
    for i in range(1, M + 1):
        wave += sine_wave(i * freq, amplitude = (1.0 / i))
    return wave * (-2 * amplitude / numpy.pi)

def sine_wave(freq, duration = 1.0, amplitude = 1.0, phase = 0.0):
    t = numpy.arange(int(duration * FS))
    return amplitude * numpy.sin(t * (freq / FS) * (2 * numpy.pi) + phase)

def db_spectrum(wave, window, zeropadding = 1):
    N = zeropadding * len(wave)
    fft = numpy.fft.fft(wave * window, N)
    bin_freqs = numpy.arange(0, N) * FS / float(N)
    fft = fft[:N/2 + 1]
    bin_freqs = bin_freqs[:N/2 + 1]
    magfft = abs(fft) / (numpy.sum(window) / 2.0)
    epsilon = pow(10, -10)
    db = 20 * numpy.log10(magfft + epsilon)
    return db, bin_freqs

# time domain
sawtooth_wave = sawtooth(1000.0, 22)
t = numpy.linspace(0, 1, 44100)
sawtooth_perfect = 0.5 * signal.sawtooth(2 * numpy.pi * 1000 * t)      # convert data into 16-bit integer

write(output_dir_reconstructed, FS, sawtooth_wave)
write(output_dir_perfect, FS, sawtooth_perfect)

# frequency domain
sawtooth_wave_db, t2 = db_spectrum(sawtooth_wave[0:8192], numpy.hanning(8192))
sawtooth_perfect_db, t2 = db_spectrum(sawtooth_perfect[0:8192], numpy.hanning(8192))

plt.figure()
plt.plot(t, sawtooth_perfect, label="Perfect sawtooth")
plt.plot(t, sawtooth_wave, label="Reconstructed")
plt.ylabel("amplitude")
plt.xlabel("time (s)")
plt.title("Using 22 sine waves")
plt.xlim(0, 0.01)
plt.legend()
plt.show()

plt.figure()
plt.plot(t2, sawtooth_perfect_db, label="Perfect sawtooth")
plt.plot(t2, sawtooth_wave_db, label="Reconstructed")
plt.ylim(-100, 0)
plt.ylabel("amplitude (db)")
plt.xlabel("frequency (bins)")
plt.title("Using 22 sine waves")
plt.legend()
plt.show()

