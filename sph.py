from functools import partial
import numpy as np


CHUNK_SIZE = 1024


class Sph(object):

    def __init__(self, channels, framerate, sample_width, data):
        self._channels = channels
        self._framerate = framerate
        self._sample_width = sample_width
        self._data = data

    @property
    def channels(self):
        return self._channels

    @property
    def framerate(self):
        return self._framerate

    @property
    def sample_width(self):
        return self._sample_width

    @property
    def data(self):
        return self._data


def read(fname):
    # passing headers may be more robust
    with open(fname, 'rb') as f:
        header = []
        for i in range(9):
            header.append(f.readline(1024))

        samples_num = int((header[2].split(' '))[2])
        sample_width = int((header[3].split(' '))[2])
        channels = int((header[4].split(' '))[2])
        framerate = int((header[6].split(' '))[2])

        samples = np.zeros((samples_num*2,), dtype=np.int16)

        counter = 0
        for chunk in iter(partial(f.read, CHUNK_SIZE*sample_width), ""):
            if counter + len(chunk) > samples_num:
                chunk = chunk[:(samples_num-counter)*sample_width]

            for i in range(len(chunk) / sample_width):
                sample = 0
                for j in range(sample_width):
                    sample |= ord(chunk[i*sample_width+j]) << 8*(sample_width-1-j)

                samples[counter] = sample
                counter += 1

            if counter == samples_num:
                break

        return Sph(channels, framerate, sample_width, samples)

