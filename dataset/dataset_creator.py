import csv
from multiprocessing import Pool, Manager

from mfcc import get_mel_filterbanks

from dataset.file_processing import process_file, create_table_header, write_features
from dataset.utils import scale_features

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath('../'))
from config import *

counter_queue = Manager().Queue(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Specify dataset. Possible: musan, tedlium. Default: musan.', default=MUSAN_DATASET)

    args = parser.parse_args()
    dataset_name = args.d

    # Pool for processing files
    pool = Pool(PROCESSES_NUM)

    # Create filterbank for processing files
    # It takes a lot of time so we pre-compute it
    fbank = get_mel_filterbanks(LOW_HZ, HIGH_HZ, FFT_N, FILTERBANKS_NUM, SAMPLERATE)

    if dataset_name == MUSAN_DATASET:
        speech_path = MUSAN_SPEECH_PATH
        noise_path = MUSAN_NOISE_PATH
        dataset_path = MUSAN_PATH
        transcription_path = None

    elif dataset_name == TEDLIUM_DATASET:
        speech_path = TEDLIUM_SPEECH_PATH
        noise_path = None
        dataset_path = TEDLIUM_PATH
        transcription_path = TEDLIUM_TRANSCRIPTION_PATH

    else:
        raise NameError('Wrong dataset name')

    voiced_csv_name = dataset_path + '/dataset/voiced.csv'
    unvoiced_csv_name = dataset_path + '/dataset/unvoiced.csv'
    noisy_speech_csv_name = dataset_path + '/dataset/noisy_speech.csv'

    # ##################################################################################
    # Process speech files
    # ##################################################################################
    if MAX_SPEECH_FILES > 0:
        if speech_path is not None:

            voiced_writer = csv.writer(open(voiced_csv_name, 'w'), delimiter=',')
            voiced_writer.writerows([create_table_header(MFCC_NUM)])

            speech_files = []
            for files_path in speech_path:
                print(files_path)
                speech_files.extend([files_path+'/'+file for file in os.listdir(files_path)
                                     if (file.endswith('.wav') or file.endswith(".sph"))])

            if len(speech_files) > MAX_SPEECH_FILES:
                speech_files = speech_files[:MAX_SPEECH_FILES]

            speech_input = []

            for file_path in speech_files:
                file_name = file_path.split('/')[-1]
                transcription_file_name = '/' + file_name.split('.')[0] + '.stm'

                speech_input.append([
                                    file_path, FRAME_SIZE, FRAME_STEP, FFT_N, fbank, MFCC_NUM,
                                    counter_queue, transcription_path + transcription_file_name
                                    ])

            files_counter = 0
            while files_counter < len(speech_files):
                if files_counter + FILES_PER_STEP < len(speech_files):
                    features = pool.map(process_file, speech_input[files_counter:files_counter + FILES_PER_STEP])
                else:
                    features = pool.map(process_file, speech_input[files_counter:])

                features = scale_features(features)
                write_features(voiced_writer, features, VOICED)
                files_counter += FILES_PER_STEP

                del features

            del speech_input

    # ##################################################################################
    # Process noise files
    # ##################################################################################
    if MAX_NOISE_FILES > 0:
        if noise_path is not None:

            unvoiced_writer = csv.writer(open(unvoiced_csv_name, 'w'), delimiter=',')
            unvoiced_writer.writerows([create_table_header(MFCC_NUM)])

            noise_files = []
            for files_path in noise_path:
                noise_files.extend([file for file in os.listdir(noise_path) if file.endswith('.wav')])

            if len(noise_files) > MAX_NOISE_FILES:
                noise_files = noise_files[:MAX_NOISE_FILES]

            noise_input = []
            noise_input.extend([
                                    (
                                        fname, FRAME_SIZE, FRAME_STEP, FFT_N, fbank, MFCC_NUM, counter_queue,
                                        None
                                    )
                                    for fname in noise_files
                                    ])

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




