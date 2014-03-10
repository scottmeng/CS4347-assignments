import math
import numpy
from scipy import signal
import pylab as plt
from scipy.io.wavfile import write
from scipy.io.wavfile import read

import os
os.getcwd()

input_a2_dir = "a2.wav"
input_a3_dir = "a3.wav"
output_dir_without_li = "part3_without_interpolation.wav"
output_dir_with_li = "part3_with_interpolation.wav"

FS = read(input_a2_dir)[0]

input_a2 = read(input_a2_dir)[1] / 32768.
input_a3 = read(input_a3_dir)[1] / 32768.

def look_up_with_li(i, step):
	index = i * step
	floor_x = int(math.floor(index))
	ceil_x = int(math.ceil(index))
	floor_y = input_a2[floor_x]
	ceil_y = input_a2[ceil_x]
	return floor_y + (ceil_y - floor_y) * (index - floor_x)

def look_up(i, step):
	index = round(i * step)
	return input_a2[index]

def generate_wave_with_li(step, duration = 1):
	wave = numpy.zeros(FS * duration)
	for i in range(FS * duration):
		wave[i] = look_up_with_li(i, step)
	return wave

def generate_wave_without_li(step, duration = 1):
	wave = numpy.zeros(FS * duration)
	for i in range(FS * duration):
		wave[i] = look_up(i, step)
	return wave

def cal_step(index):
	return pow(2, index / 6.0)

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
	plt.ylabel("Frequency (bin)")
	plt.show()
	return

data_with_li = input_a2[0:FS]
data_without_li = numpy.copy(data_with_li)

for i in range(1, 7):
	temp_with_li = generate_wave_with_li(cal_step(i))
	temp_without_li = generate_wave_without_li(cal_step(i))
	data_with_li = numpy.concatenate((data_with_li, temp_with_li))
	data_without_li = numpy.concatenate((data_without_li, temp_without_li))

data_without_li = numpy.concatenate((data_without_li, input_a3[0:FS]))
data_with_li = numpy.concatenate((data_with_li, input_a3[0:FS]))

data_with_li = numpy.array(data_with_li * 32767, dtype = numpy.int16)
data_without_li = numpy.array(data_without_li * 32767, dtype = numpy.int16)

write(output_dir_with_li, FS, data_with_li)
write(output_dir_without_li, FS, data_without_li)

show_spectrogram(data_with_li, "Output with linear interpolation")
show_spectrogram(data_without_li, "Output without linear interpolation")


