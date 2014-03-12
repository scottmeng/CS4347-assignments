import math
import numpy
import scipy.fftpack
from scipy.io.wavfile import read

import os
os.getcwd()                                              # get the directory of current process

input_dir = "music_speech.mf"
output_dir = "Assignment7.arff"
num_windows = 26

def apply_pre_emphasis(data):
    copy = numpy.delete(data, len(data) - 1)
    copy = 0.95 * numpy.insert(copy, [0.0], 0)
    return data - copy

def db_spectrum(data, window):
    fft = numpy.fft.fft(data * window)
    fft = fft[:len(fft) / 2 + 1]
    magfft = abs(fft) / (numpy.sum(window) / 2.0)
    #epsilon = pow(10, -10)
    #db = 20*numpy.log10(magfft + epsilon)
    #return db
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
    else:
        return window[0:513]

input_file = open(input_dir)                                # read from ground truth file
output_file = open(output_dir, 'w')                         # write into output ARFF file
output_file.write('@RELATION music_speech\n')
output_file.write('@ATTRIBUTE WIN1_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN2_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN3_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN4_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN5_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN6_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN7_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN8_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN9_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN10_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN11_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN12_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN13_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN14_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN15_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN16_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN17_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN18_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN19_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN20_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN21_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN22_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN23_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN24_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN25_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN26_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE WIN1_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN2_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN3_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN4_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN5_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN6_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN7_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN8_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN9_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN10_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN11_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN12_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN13_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN14_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN15_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN16_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN17_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN18_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN19_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN20_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN21_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN22_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN23_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN24_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN25_STD NUMERIC\n')
output_file.write('@ATTRIBUTE WIN26_STD NUMERIC\n')
output_file.write('@ATTRIBUTE class {music,speech}\n\n')
output_file.write('@DATA\n')

file_names = input_file.readlines()

for file_name in file_names:
    music_dir = file_name.split()[0]                     # extract music and speech file directory
    label = file_name.split()[1]
    samp_rate, data = read(music_dir)

    mel_interval = cal_mel_interval(samp_rate)
    bin_val = samp_rate / 1024.0

    windows = numpy.zeros((num_windows, 513))
    for i in range(num_windows):
        windows[i] = triangle_window(reverse_mel(i * mel_interval) / bin_val, reverse_mel((i + 1) * mel_interval) / bin_val, reverse_mel((i + 2) * mel_interval) / bin_val)

    data = data / 32768.                                 # normalize wav file data
    data = apply_pre_emphasis(data)

    num_buffers = len(data) / 512 - 1                    # calculate the number of buffer slices
    buffers = numpy.zeros((513, num_buffers))           # create matrix to store buffer data
        
    for i in range(num_buffers):                         # put buffer slices in matrix
        start = int(i * 512)
        end = int(i * 512 + 1024)
        buffer_data = data[start:end]

        buffers[:,i] = db_spectrum(buffer_data, numpy.hamming(1024))

    result = numpy.dot(windows, buffers)
    result = numpy.log10(result)
    result = scipy.fftpack.dct(result)

    means = numpy.mean(result, axis = 1)
    std_dev = numpy.std(result, axis = 1)

    means_string = numpy.char.mod('%f', means)             # generate comma seperated vector output
    means_string = ",".join(means_string)

    std_dev_string = numpy.char.mod('%f', std_dev)
    std_dev_string = ",".join(std_dev_string)

    output_file.write("%s,%s,%s\n" % (means_string, std_dev_string, label))

    print "*"  

output_file.close()      
