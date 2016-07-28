import csv
import os
from multiprocessing import Pool

from scipy.io import wavfile

from mfcc import get_mel_filterbanks, get_mfcc, get_deltas
from utils import create_table_header, split_into_frames, get_frames_energies, write_features

PROCESSES_NUM = 4
MAX_SPEECH_FILES = 9999999
MAX_NOISE_FILES = 9999999

# To prevent memory overflow
FILES_PER_STEP = 30

# Set your path
MUSAN_PATH = '/home/kript0n/Documents/musan'
SPEECH_PATH = MUSAN_PATH + '/speech'
NOISE_PATH = MUSAN_PATH + '/noise'

FRAME_SIZE = 400
FRAME_STEP = 160
LOW_HZ = 300
HIGH_HZ = 8000
MFCC_NUM = 13
FFT_N = 512

NONE_VOICED = 0
VOICED = 1


def process_file(fname):
    sample_rate, f_raw = wavfile.read(fname)

    frames = split_into_frames(f_raw, FRAME_SIZE, FRAME_STEP)
    frames_energies = get_frames_energies(frames)

    mel_filterbank = get_mel_filterbanks(LOW_HZ, HIGH_HZ, FFT_N, MFCC_NUM, sample_rate)

    frames_buffer_size = 5
    frames_buffer = []
    # middle of the buffer
    processing_frame_index = 2

    features = []

    for frame, energy in zip(frames, frames_energies):
        if len(frames_buffer) < frames_buffer_size:
            frames_buffer.append(get_mfcc(frame, FFT_N, mel_filterbank))
        else:
            processing_frame_mfcc = frames_buffer[processing_frame_index]

            prev_frame_mfcc = frames_buffer[processing_frame_index-1]
            prev_frame_first_deltas = get_deltas(processing_frame_mfcc, frames_buffer[processing_frame_index-2])

            next_frame_mfcc = frames_buffer[processing_frame_index+1]
            next_frame_first_deltas = get_deltas(frames_buffer[processing_frame_index+2], processing_frame_mfcc)

            processing_frame_first_deltas = get_deltas(next_frame_mfcc, prev_frame_mfcc)
            processing_frame_second_deltas = get_deltas(next_frame_first_deltas, prev_frame_first_deltas)

            features.append((
                processing_frame_mfcc,
                processing_frame_first_deltas,
                processing_frame_second_deltas
            ))

            # Update circular buffer
            frames_buffer.pop(0)
            frames_buffer.append(get_mfcc(frame, FFT_N, mel_filterbank))

    return features


# For you Edward
def scale_features(features):
    pass


if __name__ == '__main__':
    pool = Pool(PROCESSES_NUM)

    writer = csv.writer(open('blank.csv', 'w'), delimiter=',')
    writer.writerows([create_table_header(MFCC_NUM)])

    speech_files = [SPEECH_PATH + '/' + file for file in os.listdir(SPEECH_PATH) if file.endswith('.wav')]
    noise_files = [NOISE_PATH + '/' + file for file in os.listdir(NOISE_PATH) if file.endswith('.wav')]

    if len(speech_files) > MAX_SPEECH_FILES:
        speech_files = speech_files[:MAX_SPEECH_FILES]

    if len(noise_files) > MAX_NOISE_FILES:
        noise_files = noise_files[:MAX_NOISE_FILES]

    files_counter = 0

    while files_counter < len(speech_files):
        if files_counter + FILES_PER_STEP < len(speech_files):
            features = pool.map(process_file, speech_files[files_counter:files_counter + FILES_PER_STEP])
        else:
            features = pool.map(process_file, speech_files[files_counter:])

        write_features(writer, features, VOICED)
        files_counter += FILES_PER_STEP

    files_counter = 0

    while files_counter < len(noise_files):
        if files_counter + FILES_PER_STEP < len(noise_files):
            features = pool.map(process_file, noise_files[files_counter:files_counter + FILES_PER_STEP])
        else:
            features = pool.map(process_file, noise_files[files_counter:])

        write_features(writer, features, NONE_VOICED)
        files_counter += FILES_PER_STEP





