import csv
import os
from functools import partial
from multiprocessing import Manager

import numpy as np
from scipy.io import wavfile

from mfcc import get_mfcc, get_deltas

counter_queue = Manager().Queue(1)
counter_queue.put(0)


def create_pool_input(files_path, frame_size, frame_step, fft_n, fbank, mfcc_num, transcription_path=None):
    #
    #   Create list of input for multiprocessing.Pool.map
    #
    global counter_queue

    if transcription_path is not None:
        transcription_path += '/' + file
    input = []


    return input


def process_file(args):
    fname = args[0]
    frame_size = args[1]
    frame_step = args[2]
    fft_n = args[3]
    mel_filterbank = args[4]
    mfcc_num = args[5]
    counter_queue = args[6]
    transcription_path = args[7]

    sample_rate, f_raw = wavfile.read(fname)
    frames = split_into_frames(f_raw, frame_size, frame_step, transcription_path, sample_rate)

    frames_buffer_size = 5
    frames_buffer = []
    # middle of the buffer
    processing_frame_index = 2

    features = []

    for frame in frames:
        if len(frames_buffer) < frames_buffer_size:
            frames_buffer.append(get_mfcc(frame, fft_n, mel_filterbank, mfcc_num))
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
            frames_buffer.append(get_mfcc(frame, fft_n, mel_filterbank, mfcc_num))

    processed_files = counter_queue.get() + 1
    if processed_files % 5 == 0:
        print("Processed " + str(processed_files) + ' files')
    counter_queue.put(processed_files)

    return features


def split_into_frames(data, frame_size, step, transcription_path=None, frame_rate=None):
    #
    #   Split data into frames with given size and step
    #
    frames = []
    offset = 0

    if transcription_path and frame_rate:
        indices = parse_transcription(transcription_path, frame_rate)
        data = data[indices]

    elif transcription_path and frame_rate is None:
        raise Exception('You must specify frame_rate')

    while len(data) - offset > frame_size:
        frames.append(data[offset:offset+frame_size])
        offset += step

    return frames


def parse_transcription(path, frame_rate):
    #
    #   Parse transcription of given audio file
    #
    indices = np.array([], dtype=np.int32)

    with open(path) as f_trans:
        for line in iter(partial(f_trans.readline, 2048)):
            pass

    return indices


def create_table_header(mfcc_len):
    #
    #   Create table header for dataset csv file
    #
    header = []

    for i in range(mfcc_len):
        header.append('MFCC Coef' + str(i + 1))

    for i in range(mfcc_len):
        header.append('First delta' + str(i + 1))

    for i in range(mfcc_len):
        header.append('Second delta' + str(i + 1))

    header.append('voiced')

    return header


def write_features(writer, features, label):
    #
    #   Write features into csv file
    #
    #   Parameters:
    #       writer -- csv writer object
    #       features -- features list [[f1, f2, f3, ...], []]
    #       label -- class labels [l1, l2, ...]
    #

    for file_features in features:
        rows_to_write = []

        for i in range(len(file_features)):
            frame_features = file_features[i]
            row = np.concatenate((frame_features[0], frame_features[1], frame_features[2], [label]))
            rows_to_write.append(row)

        writer.writerows(rows_to_write)


def load_csv(fname, items_num, columns_used=None, memmap=False, dtype=np.float32):
    #
    #   Load csv file with features. Last column is treated as Label.
    #
    #   Parameters:
    #       fname -- file name
    #       items_num -- number of
    #       columns_used -- specify indices of used features
    #       memmap -- if True then numpy array is mapped onto disk (not supported yet)
    #
    with open(fname, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header = reader.next()

        if columns_used:
            feature_dim = len(columns_used)
            columns_used = np.asarray(columns_used)
        else:
            feature_dim = len(header)-1

        features = np.zeros((items_num, feature_dim), dtype=dtype)
        results = np.zeros((items_num, 1), dtype=np.int32)

        for i in range(items_num):
            line = np.asarray(reader.next(), dtype=dtype)

            if columns_used:
                np.put(features[i], np.arange(0, feature_dim), line[columns_used])
            else:
                np.put(features[i], np.arange(0, feature_dim), line[:-1])

            results[i] = int(line[-1])

        return features, results