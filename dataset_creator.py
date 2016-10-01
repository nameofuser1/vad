import csv
from multiprocessing import Pool, Manager

from dataset.file_processing import process_file, create_table_header, write_features
from dataset.utils import scale_features

import os
import argparse

from config import *
from mfcc import get_mel_filterbanks

counter_queue = Manager().Queue(1)

# precomputed filterbank for extraction MFCC
fbank = get_mel_filterbanks(LOW_HZ, HIGH_HZ, FFT_N, FILTERBANKS_NUM, SAMPLERATE)


def process_files(files_paths, files_type, max_files, csv_writer, transcription_dir=None):
    """
    Parameters
    ----------
    files_paths         --- list of directories of audio files to be processed
    files_type          --- voiced, unvoiced, music
    max_files           --- no more than max_files files will be processed
    transcription_path  --- path to transcription stm file
    csv_writer          --- csv object for saving result

    """

    # Getting all the files from given directories
    print("Creating files list...")
    files2process = []

    for files_path in files_paths:
        files2process.extend([files_path + '/' + file for file in os.listdir(files_path)
                              if (file.endswith('.wav') or file.endswith('.sph'))])

    if len(files2process) > max_files:
        files2process = files2process[:max_files]

    print("Need to process %d files" % len(files2process))

    # Create input for pool.map called on process_file
    pool_input = []
    for file_path in files2process:
        file_name = file_path.split('/')[-1]

        if transcription_dir is not None:
            transcription_file_name = '/' + file_name.split('.')[0] + '.stm'
            transcription_path = transcription_dir + transcription_file_name

        pool_input.append([
            file_path, FRAME_SIZE, FRAME_STEP, FFT_N, fbank, MFCC_NUM,
            counter_queue, transcription_path
        ])

    # Processing files
    print("Start processing files...")
    files_counter = 0
    while files_counter < len(files2process):
        if files_counter + FILES_PER_STEP < len(files2process):
            features = pool.map(process_file, pool_input[files_counter:files_counter + FILES_PER_STEP])
        else:
            features = pool.map(process_file, pool_input[files_counter:])

        features = scale_features(features)
        write_features(csv_writer, features, files_type)
        files_counter += FILES_PER_STEP

        del features

    print("All files are done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Specify dataset. Possible: musan, tedlium. Default: musan.', default=MUSAN_DATASET)

    args = parser.parse_args()
    dataset_name = args.d

    # Pool for processing files
    pool = Pool(PROCESSES_NUM)

    # 0 processed files
    counter_queue.put(0)

    # Create filterbank for processing files
    # It takes a lot of time so we pre-compute it


    if dataset_name == MUSAN_DATASET:
        speech_path = MUSAN_SPEECH_PATH
        noise_path = MUSAN_NOISE_PATH
        music_path = MUSAN_MUSIC_PATH
        dataset_path = MUSAN_PATH
        transcription_path = None

    elif dataset_name == TEDLIUM_DATASET:
        speech_path = TEDLIUM_SPEECH_PATH
        noise_path = TEDLIUM_NOISE_PATH
        music_path = TEDLIUM_MUSIC_PATH
        dataset_path = TEDLIUM_PATH
        transcription_path = TEDLIUM_TRANSCRIPTION_PATH

    else:
        raise NameError('Wrong dataset name')

    voiced_csv_name = dataset_path + '/dataset/voiced.csv'
    unvoiced_csv_name = dataset_path + '/dataset/unvoiced.csv'
    music_csv_name = dataset_path + '/dataset/music.csv'
    noisy_speech_csv_name = dataset_path + '/dataset/noisy_speech.csv'

    print("Voiced frames path: " + str(voiced_csv_name))
    print("Unvoiced frames path: " + str(unvoiced_csv_name))
    print("Noisy frames path: " + str(noise_path))

    # ##################################################################################
    # Process speech files
    # ##################################################################################
    if MAX_SPEECH_FILES > 0:
        if speech_path is not None:
            print("Begin processing speech files...")

            voiced_writer = csv.writer(open(voiced_csv_name, 'w'), delimiter=',')
            voiced_writer.writerows([create_table_header(MFCC_NUM)])

            process_files(speech_path, VOICED, MAX_SPEECH_FILES, voiced_writer, transcription_path)

    # ##################################################################################
    # Process noise files
    # ##################################################################################
    if MAX_NOISE_FILES > 0:
        if noise_path is not None:
            print("Begin processing noise files...")

            unvoiced_writer = csv.writer(open(unvoiced_csv_name, 'w'), delimiter=',')
            unvoiced_writer.writerows([create_table_header(MFCC_NUM)])

            process_files(noise_path, NONE_VOICED, MAX_NOISE_FILES, unvoiced_writer, None)

    # ##################################################################################
    # Process music files
    # ##################################################################################
    if MAX_MUSIC_FILES > 0:
        if music_path is not None:
            print("Begin processing music files...")

            music_writer = csv.writer(open(music_csv_name, 'w'), delimiter=',')
            music_writer.writerows([create_table_header(MFCC_NUM)])

            process_files(music_path, MUSIC, MAX_MUSIC_FILES, music_writer, None)

    print("Yeeeeaaahhh")


