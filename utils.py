import numpy as np


def get_energy_threshold(frames_energies):
    mean = np.mean(frames_energies)
    return np.mean(frames_energies[np.where(frames_energies < mean)])


def split_into_frames(data, frame_size, step):
    frames = []
    offset = 0

    while len(data) - offset > frame_size:
        frames.append(data[offset:offset+frame_size])
        offset += step

    return frames


def get_energy(frame):
    return np.sum(np.square(frame)) / np.float64(len(frame))


def get_frames_energies(frames):
    en = np.array([])

    for frame in frames:
        en = np.append(en, get_energy(frame))

    return en


def create_table_header(mfcc_len):
    header = []

    for i in range(mfcc_len):
        header.append('MFCC Coef' + str(i + 1))

    for i in range(mfcc_len):
        header.append('First delta' + str(i + 1))

    for i in range(mfcc_len):
        header.append('Second delta' + str(i + 1))

    header.append('voiced')

    return header


def write_features(writer, features, voiced):

    for frames_features in features:
        rows_to_write = []

        for i in range(len(frames_features)):
            frame_features = frames_features[i]
            row = np.concatenate((frame_features[0], frame_features[1], frame_features[2], [voiced]))
            rows_to_write.append(row)

        writer.writerows(rows_to_write)


