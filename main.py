from mfcc import get_mfcc, get_deltas
import csv
import numpy as np
from functools import partial


def process_file(fname, sample_rate, frame_size, frame_step, low_hz, high_hz, n_mfcc, fft_n):
    f_raw = b''

    with open(fname, 'rb') as wav_file:
        # passing WAV headers
        wav_file.read(44)

        for chunk in iter(partial(wav_file.read, 1024), ''):
            f_raw += chunk

    # or ceil? samples number equal frame_size will give us 0 instead of 1
    frames_number = np.floor_divide((len(f_raw) - frame_size), frame_step)
    print("Frames in file " + fname + ": " + str(frames_number))

    frames_buffer_size = 5
    frames_buffer = []
    # middle of the buffer
    processing_frame_index = 2

    features = []

    offset = 0
    for i in range(frames_number):
        frame = np.frombuffer(f_raw[offset:offset+frame_size], dtype=np.int8)
        offset += frame_step

        if len(frames_buffer) < frames_buffer_size:
            frames_buffer.append(get_mfcc(fft_n, frame, n_mfcc, low_hz, high_hz, sample_rate))
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
            frames_buffer.append(get_mfcc(fft_n, frame, n_mfcc, low_hz, high_hz, sample_rate))

            if i % 100 == 0:
                print('Processed ' + str(i) + ' frames of ' + str(frames_number))

    return features


def create_table_header(mfcc_len, first_deltas_len, second_deltas_len):
    header = []

    for i in range(mfcc_len):
        header.append('MFCC Coef' + str(i + 1))

    for i in range(first_deltas_len):
        header.append('First delta' + str(i + 1))

    for i in range(second_deltas_len):
        header.append('Second delta' + str(i + 1))

    header.append('voiced')

    return header


if __name__ == '__main__':
    writer = csv.writer(open('blank.csv', 'w'), delimiter=',')
    frames_features = process_file('./music.wav', 16000, 400, 160, 300, 8000, 13, 512)
    header = []

    print(frames_features[0])

    mfcc_vec_len = len(frames_features[0][0])
    first_deltas_vec_len = len(frames_features[0][1])
    second_deltas_vec_len = len(frames_features[0][2])

    rows_to_write = [create_table_header(mfcc_vec_len, first_deltas_vec_len, second_deltas_vec_len)]

    for i in range(len(frames_features)):
        frame_features = frames_features[i]
        row = np.concatenate((frame_features[0], frame_features[1], frame_features[2], [1]))
        rows_to_write.append(row)

    writer.writerows(rows_to_write)
