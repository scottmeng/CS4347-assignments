import math
import numpy
from scipy import signal
import pylab as plt
from scipy.io.wavfile import write

import os
os.getcwd()

FS = 44100
N_high = 16384
N_low = 2048

#sine_high = numpy.sin(2 * numpy.pi * numpy.linspace(0, 1, N_high))
#sine_low = numpy.sin(2 * numpy.pi * numpy.linspace(0, 1, N_low))

t = numpy.arange(N_high)
sine_high = numpy.sin(t * (1.0 / N_high) * (2 * numpy.pi))
t = numpy.arange(N_low)
sine_low = numpy.sin(t * (1.0 / N_low) * (2 * numpy.pi)) 

def sine_wave(freq, duration = 1.0, amplitude = 1.0, phase = 0.0):
    t = numpy.arange(int(duration * FS))
    return amplitude * numpy.sin(t * (freq / FS) * (2 * numpy.pi) + phase)

def LUT(i, freq, no_sample):
	index = round(i * freq * no_sample / FS) % no_sample
	if no_sample == N_high:
		return sine_high[index]
	if no_sample == N_low:
		return sine_low[index]

def linear_interpolation(i, freq, no_sample):
	index = (i * freq * no_sample / FS) % no_sample
	floor_x = int(math.floor(index)) % no_sample
	ceil_x = int(math.ceil(index)) % no_sample

	if no_sample == N_high:
		floor_y = sine_high[floor_x]
		ceil_y = sine_high[ceil_x]
	else:
		floor_y = sine_low[floor_x]
		ceil_y = sine_low[ceil_x]
	val = floor_y + (ceil_y - floor_y) * (index - floor_x)
	return val

time = numpy.linspace(0, 1, FS)

lut_wave_high_100 = numpy.zeros(FS)
lut_wave_high_1234_56 = numpy.zeros(FS)
lut_wave_low_100 = numpy.zeros(FS) 
lut_wave_low_1234_56 = numpy.zeros(FS)

li_wave_high_100 = numpy.zeros(FS)
li_wave_high_1234_56 = numpy.zeros(FS)
li_wave_low_100 = numpy.zeros(FS) 
li_wave_low_1234_56 = numpy.zeros(FS)

# generate wave using look up table with no interpolation
for i in xrange(FS):
	lut_wave_high_100[i] = LUT(i, 100.0, N_high)
	lut_wave_high_1234_56[i] = LUT(i, 1234.56, N_high)
	lut_wave_low_100[i] = LUT(i, 100, N_low)
	lut_wave_low_1234_56[i] = LUT(i, 1234.56, N_low)
	li_wave_high_100[i] = linear_interpolation(i, 100.0, N_high)
	li_wave_high_1234_56[i] = linear_interpolation(i, 1234.56, N_high)
	li_wave_low_100[i] = linear_interpolation(i, 100.0, N_low)
	li_wave_low_1234_56[i] = linear_interpolation(i, 1234.56, N_low)

perfect_sine_wave_100 = sine_wave(100.0)
perfect_sine_wave_1234_56 = sine_wave(1234.56)

lut_max_error_low_100 = numpy.max(numpy.abs(lut_wave_low_100 - perfect_sine_wave_100))
lut_max_error_high_100 = numpy.max(numpy.abs(lut_wave_high_100 - perfect_sine_wave_100))
lut_max_error_low_1234_56 = numpy.max(numpy.abs(lut_wave_low_1234_56 - perfect_sine_wave_1234_56))
lut_max_error_high_1234_56 = numpy.max(numpy.abs(lut_wave_high_1234_56 - perfect_sine_wave_1234_56))

li_max_error_low_100 = numpy.max(numpy.abs(li_wave_low_100 - perfect_sine_wave_100))
li_max_error_high_100 = numpy.max(numpy.abs(li_wave_high_100 - perfect_sine_wave_100))
li_max_error_low_1234_56 = numpy.max(numpy.abs(li_wave_low_1234_56 - perfect_sine_wave_1234_56))
li_max_error_high_1234_56 = numpy.max(numpy.abs(li_wave_high_1234_56 - perfect_sine_wave_1234_56))

print "=== Look up table with no linear interpolation ==="
print "2048 Samples 100 Hz: "
print lut_max_error_low_100 * 32767
print "2048 Samples 1234.56 Hz: "
print lut_max_error_low_1234_56 * 32767

print "16384 Samples 100 Hz: "
print lut_max_error_high_100 * 32767
print "16384 Samples 1234.56 Hz: "
print lut_max_error_high_1234_56 * 32767

print "=== Look up table with linear interpolation ==="
print "2048 Samples 100 Hz: "
print li_max_error_low_100 * 32767
print "2048 Samples 1234.56 Hz: "
print li_max_error_low_1234_56 * 32767

print "16384 Samples 100 Hz: "
print li_max_error_high_100 * 32767
print "16384 Samples 1234.56 Hz: "
print li_max_error_high_1234_56 * 32767