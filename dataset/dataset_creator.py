import csv
from multiprocessing import Pool

from dataset.file_processing import process_file, create_table_header, write_features, create_pool_input
from dataset.utils import scale_features

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath('../'))
from config import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Specify dataset. Possible: musan, tedlium. Default: musan.', const=MUSAN_DATASET)

    args = parser.parse_args()
    dataset_name = args.d

    if dataset_name == MUSAN_DATASET:
        speech_path = MUSAN_SPEECH_PATH
        noise_path = MUSAN_NOISE_PATH

    elif dataset_name == TEDLIUM_DATASET:
        speech_path = TEDLIUM_SPEECH_PATH
        noise_path = TEDLIUM_NOISE_PATH

    else:
        raise NameError('Wrong dataset name')

    # Override files only if we have something to write
    if MAX_SPEECH_FILES > 0:
        voiced_writer = csv.writer(open(VOICED_FNAME, 'w'), delimiter=',')
        voiced_writer.writerows([create_table_header(MFCC_NUM)])

    if MAX_NOISE_FILES > 0:
        unvoiced_writer = csv.writer(open(UNVOICED_FNAME, 'w'), delimiter=',')
        unvoiced_writer.writerows([create_table_header(MFCC_NUM)])

    # Get files names
    speech_files = []
    for files_path in speech_path:
        speech_files.extend([file for file in os.listdir(files_path) if file.endswith('.wav')])

    noise_files = []
    for files_path in noise_path:
        noise_files.extend([file for file in os.listdir(noise_path) if file.endswith('.wav')])

    # Create input for pool.map
    input = []
    input.extend([
                     (
                         file, frame_size, frame_step, fft_n, fbank, mfcc_num, counter_queue,
                         transcription_path
                     )
                     for
                     ])

    if len(speech_files) > MAX_SPEECH_FILES:
        speech_files = speech_files[:MAX_SPEECH_FILES]

    if len(noise_files) > MAX_NOISE_FILES:
        noise_files = noise_files[:MAX_NOISE_FILES]


    pool = Pool(PROCESSES_NUM)

    files_counter = 0
    while files_counter < len(speech_files):
        if files_counter + FILES_PER_STEP < len(speech_files):
            features = pool.map(process_file, speech_files[files_counter:files_counter + FILES_PER_STEP])
        else:
            features = pool.map(process_file, speech_files[files_counter:])

        features = scale_features(features)
        write_features(voiced_writer, features, VOICED)
        files_counter += FILES_PER_STEP
    del features

    files_counter = 0
    while files_counter < len(noise_files):
        if files_counter + FILES_PER_STEP < len(noise_files):
            features = pool.map(process_file, noise_files[files_counter:files_counter + FILES_PER_STEP])
        else:
            features = pool.map(process_file, noise_files[files_counter:])

        features = scale_features(features)
        write_features(unvoiced_writer, features, NONE_VOICED)
        files_counter += FILES_PER_STEP
    del features




