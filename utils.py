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

    for file_features in features:
        rows_to_write = []

        for i in range(len(file_features)):
            frame_features = file_features[i]
            row = np.concatenate((frame_features[0], frame_features[1], frame_features[2], [voiced]))
            rows_to_write.append(row)

        writer.writerows(rows_to_write)


def unshared_copy(inlist):
    if isinstance(inlist, list):
        return list(map(unshared_copy, inlist))
    elif isinstance(inlist, np.ndarray):
        return np.array(map(unshared_copy, inlist))

    return inlist


# For you Edward
def scale_features(features):
    scaled_features = features #unshared_copy(features)
    mfcc_buffer = []
    delta1_buffer = []
    delta2_buffer = []

    for file_features in scaled_features:
        for frame_features in file_features:
            mfcc_buffer.append(frame_features[0])
            delta1_buffer.append(frame_features[1])
            delta2_buffer.append(frame_features[2])

    mfcc_mean = np.mean(mfcc_buffer)
    delta1_mean = np.mean(delta1_buffer)
    delta2_mean = np.mean(delta2_buffer)

    mfcc_std = np.std(mfcc_buffer)
    delta1_std = np.std(delta1_buffer)
    delta2_std = np.std(delta2_buffer)

    for file_features in scaled_features:
        for frame_features in file_features:
            for i in range(len(frame_features[0])):
                frame_features[0][i] = (frame_features[0][i] - mfcc_mean) / mfcc_std
                frame_features[1][i] = (frame_features[1][i] - delta1_mean) / delta1_std
                frame_features[2][i] = (frame_features[2][i] - delta2_mean) / delta2_std

    return scaled_features

