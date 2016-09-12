from multiprocessing import Manager
import sounddevice as sd
import numpy as np

samples_queue = Manager().Queue()


def cb(in_data, frames, time, status):
    if frames > 0:
        samples_queue.put(np.fromstring(in_data, np.int8))


class Recoder:

    def __init__(self, frame_rate, period, channels, dtype='int16'):
        #
        #   Period in ms
        #
        self.proc = None
        self.frame_rate = frame_rate
        self.channels = channels
        self.chunk_size = int(frame_rate * period)

        self.stream = sd.RawInputStream(samplerate=self.frame_rate, blocksize=self.chunk_size,
                                   channels=self.channels, callback=cb, dtype=dtype)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.close()

    def empty(self):
        return samples_queue.empty()

    def read(self):
        return samples_queue.get()