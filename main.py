import csv
import os
import numpy as np
from scipy.io import wavfile

from mfcc import get_mel_filterbanks, get_mfcc, get_deltas
from utils import create_table_header, split_into_frames, get_frames_energies

from multiprocessing import Pool, Lock

PROCESSES_NUM = 4

MUSAN_PATH = '/home/kript0n/Documents/musan'
SPEECH_PATH = MUSAN_PATH + '/speech'
NOISE_PATH = MUSAN_PATH + '/noise'

FRAME_SIZE = 400
FRAME_STEP = 160
LOW_HZ = 300
HIGH_HZ = 8000
MFCC_NUM = 13
FFT_N = 512


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


if __name__ == '__main__':
    writer = csv.writer(open('blank.csv', 'w'), delimiter=',')
    writer.writerows([create_table_header(MFCC_NUM)])

    speech_files = [SPEECH_PATH + '/' + file for file in os.listdir(SPEECH_PATH) if file.endswith('.wav')]

    pool = Pool(PROCESSES_NUM)
    features = pool.map(process_file, speech_files)

    for frames_features in features:
        mfcc_vec_len = len(frames_features[0][0])
        first_deltas_vec_len = len(frames_features[0][1])
        second_deltas_vec_len = len(frames_features[0][2])

        rows_to_write = []

        for i in range(len(frames_features)):
            frame_features = frames_features[i]
            row = np.concatenate((frame_features[0], frame_features[1], frame_features[2], [1]))
            rows_to_write.append(row)

        writer.writerows(rows_to_write)
