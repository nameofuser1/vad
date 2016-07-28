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
