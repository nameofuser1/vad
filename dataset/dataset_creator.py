import csv
from multiprocessing import Pool, Manager

from mfcc import get_mel_filterbanks

from dataset.file_processing import process_file, create_table_header, write_features, create_pool_input
from dataset.utils import scale_features

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath('../'))
from config import *

counter_queue = Manager().Queue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Specify dataset. Possible: musan, tedlium. Default: musan.', const=MUSAN_DATASET)

    args = parser.parse_args()
    dataset_name = args.d

    if dataset_name == MUSAN_DATASET:
        speech_path = MUSAN_SPEECH_PATH
        noise_path = MUSAN_NOISE_PATH
        transcription_path = None

    elif dataset_name == TEDLIUM_DATASET:
        speech_path = TEDLIUM_SPEECH_PATH
        noise_path = None
        transcription_path = TEDLIUM_TRANSCRIPTION_PATH

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

    if len(speech_files) > MAX_SPEECH_FILES:
        speech_files = speech_files[:MAX_SPEECH_FILES]

    if len(noise_files) > MAX_NOISE_FILES:
        noise_files = noise_files[:MAX_NOISE_FILES]

    # Create filterbank for processing files
    # It takes a lot of time so we precompute it
    fbank = get_mel_filterbanks(LOW_HZ, HIGH_HZ, FFT_N, FILTERBANKS_NUM, SAMPLERATE)

    # Create input for pool.map
    speech_input = []
    speech_input.extend([
                     (
                         fname, FRAME_SIZE, FRAME_STEP, FFT_N, fbank, MFCC_NUM, counter_queue,
                         transcription_path
                     )
                     for fname in speech_files
                     ])

    # Pool for processing files
    pool = Pool(PROCESSES_NUM)

    # Process speech files
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

    del speech_input


    noise_input = []


    # Process noise files
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




