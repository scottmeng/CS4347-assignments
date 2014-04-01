import math
import numpy
import scipy.fftpack
import pylab as plt
from scipy.io.wavfile import read

import os
os.getcwd()                                              # get the directory of current process

input_dir = "../music_speech.mf"
output_dir_regular = "Assignment10_regular.arff"
output_dir_normalized = "Assignment10_normalized.arff"
num_windows = 26

def apply_pre_emphasis(data):
    copy = numpy.delete(data, len(data) - 1)
    copy = 0.95 * numpy.insert(copy, [0.0], 0)
    return data - copy

def fft_spectrum(data, window):
    fft = numpy.fft.fft(data * window)
    fft = fft[:len(fft) / 2 + 1]
    magfft = abs(fft) / (numpy.sum(window) / 2.0)
    return magfft

def cal_mel_interval(fs):
    nyquist_freq = fs / 2.0;
    return mel(nyquist_freq) / (num_windows + 1)

def reverse_mel(mel_val):
    return (math.exp(mel_val / 1127.0) - 1) * 700

def mel(freq):
    return 1127 * math.log(1 + freq / 700.0)

def triangle_window(left, middle, right):
    left = math.floor(left)
    middle = round(middle)
    right = math.ceil(right)
    ascending_half = numpy.delete(numpy.linspace(0, 1, num = (middle - left + 1)), middle - left)
    descending_half = numpy.linspace(1, 0, num = (right - middle + 1))
    window = numpy.concatenate((numpy.zeros(left), ascending_half, descending_half))
    if len(window) < 513:
        return numpy.concatenate((window, numpy.zeros(512 - right)))
    return window[0:513]

def output_arff_title(output_file):
    output_file.write('@RELATION music_speech\n')
    for i in range(num_windows):
        output_file.write('@ATTRIBUTE WIN%d_MEAN NUMERIC\n' % (i))
    for i in range(num_windows):
        output_file.write('@ATTRIBUTE WIN%d_STD NUMERIC\n' % (i))
    output_file.write('@ATTRIBUTE class {music,speech}\n\n')
    output_file.write('@DATA\n')

def output_arff_data(result, output_file):
    means = numpy.mean(result, axis = 1)
    std_dev = numpy.std(result, axis = 1)

    means_string = numpy.char.mod('%f', means)              # generate comma seperated vector output
    means_string = ",".join(means_string)

    std_dev_string = numpy.char.mod('%f', std_dev)
    std_dev_string = ",".join(std_dev_string)

    output_file.write("%s,%s,%s\n" % (means_string, std_dev_string, label))

input_file = open(input_dir)                                # read from ground truth file
output_file = open(output_dir, 'w')                         # write into output ARFF file
output_arff_title(output_file)

file_names = input_file.readlines()

for file_name in file_names:
    music_dir = file_name.split()[0]                        # extract music and speech file directory
    label = file_name.split()[1]
    samp_rate, data = read("../" + music_dir)

    mel_interval = cal_mel_interval(samp_rate)
    bin_val = samp_rate / 1024.0

    windows = numpy.zeros((num_windows, 513))
    for i in range(num_windows):
        windows[i] = triangle_window(reverse_mel(i * mel_interval) / bin_val, reverse_mel((i + 1) * mel_interval) / bin_val, reverse_mel((i + 2) * mel_interval) / bin_val)

    data = data / 32768.                                    # normalize wav file data
    data = apply_pre_emphasis(data)

    num_buffers = len(data) / 512 - 1                       # calculate the number of buffer slices
    buffers = numpy.zeros((513, num_buffers))               # create matrix to store buffer data
        
    for i in range(num_buffers):                            # put buffer slices in matrix
        start = int(i * 512)
        end = int(i * 512 + 1024)
        buffers[:,i] = fft_spectrum(data[start:end], numpy.hamming(1024))

    result = numpy.dot(windows, buffers)                    # apply 26 windows in one go
    result = numpy.log10(result)                            # apply base 10 log

    result = scipy.fftpack.dct(result, axis = 0)            # apply dct across 26 MFCC values

    output_arff_data(result, output_file)

    print "*"  

output_file.close()      
