import math
import numpy
from scipy import signal
from scipy.fftpack import fft
from scipy.io.wavfile import read

import os
os.getcwd()                                             # get the directory of current process

input_dir = "ipython/music_speech.mf"
output_dir = "ipython/assignment4.arff"

def sc_matrix(data):                                    # calculate sc for the entire matrix
    return numpy.sum(data * numpy.arange(data.shape[1]), axis = 1) / numpy.sum(data, axis = 1)

def sro_matrix(data):                                   # calculate sro for one array
    total_sum = numpy.sum(data)
    cur_sum = 0.0
    index = 0
    while(True):
        cur_sum += data[index]
        if cur_sum >= total_sum * 0.85:
            return index
        index += 1

def sfm_matrix(data):                                   # calculate sfm for the entire matrix
    return numpy.exp(numpy.mean(numpy.log(data), axis = 1)) / numpy.mean(data, axis = 1)

def par_matrix(data):                                   # calculate par for the entire matrix
    return numpy.amax(data, axis = 1) / (numpy.sqrt(numpy.mean(numpy.square(data), axis = 1)))

def sf_matrix(data):                                    # calculate sf for the entire matrix
    prev = numpy.vstack([numpy.zeros(data.shape[1]), data[:-1]])
    return numpy.sum((data - prev).clip(0), axis = 1)


input_file = open(input_dir)                            # read from ground truth file
output_file = open(output_dir, 'w')                     # write into output ARFF file
output_file.write('@RELATION music_speech\n')
output_file.write('@ATTRIBUTE SC_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE SRO_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE SFM_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE PARFFT_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE SF_MEAN NUMERIC\n')
output_file.write('@ATTRIBUTE SC_STD NUMERIC\n')
output_file.write('@ATTRIBUTE SRO_STD NUMERIC\n')
output_file.write('@ATTRIBUTE SFM_STD NUMERIC\n')
output_file.write('@ATTRIBUTE PARFFT_STD NUMERIC\n')
output_file.write('@ATTRIBUTE SF_STD NUMERIC\n')
output_file.write('@ATTRIBUTE class {music,speech}\n\n')
output_file.write('@DATA\n')

file_names = input_file.readlines()                     # read all files and directory

num_audio_files = len(file_names)                       # count the number of audio files
features_music = numpy.zeros((num_audio_files, 4))      # 2D matrix to store features from music files
features_speech = numpy.zeros((num_audio_files, 4))     # 2D matrix to store features from speech files
index = 0
features = numpy.zeros(10)                              # single vector to store all computed features

for file_name in file_names:
    music_dir, label = file_name.split()                # extract file directory and label

    data = read(music_dir)[1]
        
    data = data / 32768.                                # normalize wav file data
        
    num_buffers = len(data) / 512 - 1                   # calculate the number of buffer slices
    buffers = numpy.zeros((num_buffers, 513))           # create matrix to store buffer data
    
    for i in range(num_buffers):                        # put buffer slices in matrix
        start = int(i * 512)
        end = int(i * 512 + 1024)
        buf = data[start:end]
        buf = buf * signal.hamming(1024)                # perform windowing 
        buf = fft(buf)                                  # perform fft
        buf = buf[:len(buf) / 2 + 1]                    # extract positive portion
        buf = numpy.abs(buf)                            # take absolute
        buffers[i] = buf

    sc_buffers = sc_matrix(buffers)
    sro_buffers = numpy.apply_along_axis(sro_matrix, 1, buffers)   
    sfm_buffers = sfm_matrix(buffers)
    par_buffers = par_matrix(buffers)
    sf_buffers = sf_matrix(buffers)

    features[0] = numpy.mean(sc_buffers)
    features[1] = numpy.mean(sro_buffers)
    features[2] = numpy.mean(sfm_buffers)
    features[3] = numpy.mean(par_buffers)
    features[4] = numpy.mean(sf_buffers)
    features[5] = numpy.std(sc_buffers)
    features[6] = numpy.std(sro_buffers)
    features[7] = numpy.std(sfm_buffers)
    features[8] = numpy.std(par_buffers)
    features[9] = numpy.std(sf_buffers)    
    
    features_string = numpy.char.mod('%f', features)    # generate comma seperated vector output
    features_string = ",".join(features_string)
    
    output_file.write("%s,%s\n" % (features_string, label))
    
    print '*'

output_file.close()


