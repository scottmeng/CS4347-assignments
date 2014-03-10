import math
import numpy
from scipy import signal
import pylab as plt
from scipy.io.wavfile import write
from scipy.io.wavfile import read

import os
os.getcwd()

def karplus_strong(data):
	copy = numpy.append(data, data[0])
	copy = numpy.delete(copy, 0)
	return (copy + data) / 2.0

def db_spectrum(data, window):
	fft = numpy.fft.fft(data * window)
	fft = fft[:len(fft) / 2 + 1]
	magfft = abs(fft) / (numpy.sum(window) / 2.0)
	epsilon = pow(10, -10)
	db = 20*numpy.log10(magfft + epsilon)
	return db

def show_spectrogram(data, title):
	#data = data / 32768.
	num_buffers = len(data) / 1024 - 1                   # calculate the number of buffer slices
	buffers = numpy.zeros((num_buffers, 1025))           # create matrix to store buffer data

	for i in range(num_buffers):                         # put buffer slices in matrix
		start = int(i * 1024)
		end = int(i * 1024 + 2048)
		buf = data[start:end]
		buffers[i] = db_spectrum(buf, numpy.blackman(2048))

	buffers = buffers.transpose()
	plt.imshow(buffers, origin='lower', aspect='auto')
	plt.title(title)
	plt.xlabel("Time (buffer index)")
	plt.ylabel("Frequency (bins)")
	plt.show()
	return

FS = 32000

output_dir_400 = "part4_400.wav"
output_dir_500 = "part5_500.wav"

freq_400 = 400
freq_500 = 500

duration = 4

noiseburst_400 = 2 * numpy.random.random(FS / freq_400) - 1 
noiseburst_500 = 2 * numpy.random.random(FS / freq_500) - 1

noiserepeat_400 = numpy.zeros(FS * duration)
noiserepeat_500 = numpy.zeros(FS * duration)

for i in xrange(freq_400 * duration):
	begin = (i) * (FS / freq_400)
	end = (i+1) * (FS / freq_400)
	noiserepeat_400[begin:end] = noiseburst_400
	noiseburst_400 = karplus_strong(noiseburst_400)

for i in xrange(freq_500 * duration):
	begin = (i) * (FS / freq_500)
	end = (i+1) * (FS / freq_500)
	noiserepeat_500[begin:end] = noiseburst_500
	noiseburst_500 = karplus_strong(noiseburst_500)

# convert into int16 format
noiserepeat_400 = numpy.array(noiserepeat_400 * 32767, dtype = numpy.int16)
noiserepeat_500 = numpy.array(noiserepeat_500 * 32767, dtype = numpy.int16)

write(output_dir_400, FS, noiserepeat_400)
write(output_dir_500, FS, noiserepeat_500)

show_spectrogram(noiserepeat_400, "Pluck at 400 Hz")
show_spectrogram(noiserepeat_500, "Pluck at 500 Hz")


