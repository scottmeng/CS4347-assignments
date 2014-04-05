import math
import numpy
import scipy.fftpack
import pylab as plt
from scipy.io.wavfile import read

import os
os.getcwd()                                              # get the directory of current process

input_dir = "../music_speech.mf"
new_data_dir = "../expanded-music-speech.mf"
output_dir_regular = "Assignment10_regular_new.arff"
output_dir_normalized = "Assignment10_normalized_new.arff"
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
    for i in range(data.shape[0]):
        data_strings = numpy.char.mod('%f', data[i, :])
        data_strings = ",".join(data_strings)
        output_file.write("%s,%s\n" % (data_strings, labels[i]))

def normalize_data(data):
    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - numpy.min(data[:, i])) / (numpy.max(data[:, i]) - numpy.min(data[:, i]))
    return data

def get_mean_and_std(data):
    means = numpy.mean(result, axis = 1)
    std_devs = numpy.std(result, axis = 1)
    mean_and_std = numpy.concatenate((means, std_devs))
    return mean_and_std

def get_window(num_windows):
    windows = numpy.zeros((num_windows, 513))
    for i in range(num_windows):
        windows[i] = triangle_window(reverse_mel(i * mel_interval) / bin_val, reverse_mel((i + 1) * mel_interval) / bin_val, reverse_mel((i + 2) * mel_interval) / bin_val)
    return windows

def J48_classifier(data):
    if float(data[31]) <= 0.454318:
        if float(data[51]) <= 0.145716:
            if float(data[3]) <= 0.308147:
                return "speech"
            else:
                return "music"
        else:
            return "music"
    else:
        if float(data[13]) <= 0.667073:
            return "speech"
        else:
            return "music"

data_file = open(output_dir_normalized)
arff_entries = data_file.readlines()
start = False
num_correct_entries = 0
num_entries = 0

for arff_entry in arff_entries:
    if start:
        data_entries = arff_entry.split()[0].split(",")
        original_class = data_entries[-1]
        classified_class = J48_classifier(data_entries)
        if classified_class == original_class:
            num_correct_entries += 1
        num_entries += 1
    if arff_entry == "@DATA\n":
        start = True

print num_entries
print num_correct_entries