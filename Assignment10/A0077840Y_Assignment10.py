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

def output_arff_data(data, output_file):
    print len(labels)
    for i in range(data.shape[0]):
        data_strings = numpy.char.mod('%f', data[i, :])
        data_strings = ",".join(data_strings)
        print i
        output_file.write("%s,%s\n" % (data_strings, labels[i]))

def normalize_data(data):
    for i in range(data.shape[0]):
        data[i, :] = (data[i, :] - numpy.min(data[i, :])) / (numpy.max(data[i, :]) - numpy.min(data[i, :]))
    return data

def get_mean_and_std(data):
    means = numpy.mean(result, axis = 1)
    std_devs = numpy.std(result, axis = 1)
    mean_and_std = numpy.concatenate((means, std_devs))
    return mean_and_std

input_file = open(input_dir)                                # read from ground truth file
output_file_regular = open(output_dir_regular, 'w')         # write into output ARFF file
output_file_normalized = open(output_dir_normalized, 'w')
output_arff_title(output_file_regular)
output_arff_title(output_file_normalized)

file_names = input_file.readlines()
arff_results = numpy.empty((len(file_names), num_windows * 2))
labels = list()                                                 # list to store all the labels

for j in range(len(file_names)):
    music_dir, label = file_names[j].split()                    # extract music and speech file directory
    labels.append(label)
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

    arff_results[j] = get_mean_and_std(result)
    print "*"

output_arff_data(arff_results, output_file_regular)
arff_results_normalized = normalize_data(arff_results)
output_arff_data(arff_results_normalized, output_file_normalized)

output_file_regular.close()
output_file_normalized.close()      
