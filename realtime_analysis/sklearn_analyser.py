import cPickle
import numpy as np
from analyser import Analyser
from mfcc import get_mel_filterbanks, get_mfcc_from_spec, get_deltas, get_spec_mag, get_mfcc
from utils import first_order_low_pass

import logging

logger = logging.getLogger(__name__)
fhanlder = logging.FileHandler("analyser.log", mode='w')
logger.addHandler(fhanlder)
logger.setLevel(logging.DEBUG)


class SKLearnAnalyzer(Analyser):

    FRAMES_BUFFER_SIZE = 5
    NOISE_BUFFER_SIZE = 5
    PROCESSING_FRAME_INDEX = 2

    def __init__(self, fname, sample_rate=16000, fft_n=512, mfcc_num=13, low_hz=300, high_hz=8000, fbank_num=26):
        self.sample_rate = sample_rate
        self.fft_n = fft_n
        self.mfcc_num = mfcc_num
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.fbank_num = fbank_num

        self.filterbank = get_mel_filterbanks(low_hz, high_hz, fft_n, fbank_num, sample_rate)
        self.frames_buffer = []
        self.frames_mfcc_buffer = []
        self.noise_buffer = []

        with open(fname, 'rb') as f:
            self.classifier = cPickle.load(f)

    def load_init_inactive_frames(self, frames):
        """
        Noise frames are used for spectral subtraction
        """
        if len(frames) != SKLearnAnalyzer.NOISE_BUFFER_SIZE:
            raise ValueError("Number of inactive frame must be the same as BUFFER SIZE")

        self.noise_buffer = [get_spec_mag(frame, self.fft_n) for frame in frames]

    def feed_frame(self, frame):
        # Fill in buffer with needed number of frames
        if len(self.frames_buffer) < SKLearnAnalyzer.FRAMES_BUFFER_SIZE:
            self.__update_frames_buffers(frame)
            return None

        processing_frame = self.frames_buffer[self.PROCESSING_FRAME_INDEX]
        processing_frame_mfcc = self.frames_mfcc_buffer[self.PROCESSING_FRAME_INDEX]
        processing_frame_mfcc = self.__normalize_frame_mfcc(processing_frame_mfcc)

        prev_frame_mfcc = self.frames_mfcc_buffer[self.PROCESSING_FRAME_INDEX - 1]
        prev_frame_first_deltas = get_deltas(processing_frame_mfcc,
                                             self.frames_mfcc_buffer[self.PROCESSING_FRAME_INDEX - 2])

        next_frame_mfcc = self.frames_mfcc_buffer[self.PROCESSING_FRAME_INDEX + 1]
        next_frame_first_deltas = get_deltas(self.frames_mfcc_buffer[self.PROCESSING_FRAME_INDEX + 2],
                                             processing_frame_mfcc)

        processing_frame_first_deltas = get_deltas(next_frame_mfcc, prev_frame_mfcc)
        processing_frame_second_deltas = get_deltas(next_frame_first_deltas, prev_frame_first_deltas)

        features = np.array([], dtype=np.float32)
        features = np.append(features, (processing_frame_mfcc, processing_frame_first_deltas,
                                        processing_frame_second_deltas))

        cls = self.classifier.predict(features.reshape(1, -1))

        # add new frame
        self.__update_frames_buffers(frame)

        if cls == 1:
            return processing_frame
        elif cls == 0:
            self.__update_noise_buffer(processing_frame)
            return None
        else:
            raise AssertionError('Wrong classifier class')

    def __noise_spec_subtraction(self, spec_mag):
        """
        Performs spectral subtraction on processing frame spectrum

        Parameters
        ----------
        spec_mag    --- processing frame spectrum
        """
        logger.debug("Spectral subtraction on frame: \r\n" + str(spec_mag))
        noise_mean_spec = np.mean(self.noise_buffer, axis=0)
        noise_mean_spec = first_order_low_pass(noise_mean_spec)
        logger.debug("Spectral noise estimate: \r\n" + str(noise_mean_spec))

        subtracted = np.subtract(spec_mag, noise_mean_spec)
        subtracted = np.where(subtracted < 0, 0, subtracted)
        logger.debug("After subtraction: \r\n" + str(subtracted) + "\r\n")

        return subtracted

    def __normalize_frame_mfcc(self, mfcc):
        mfcc_mean = np.mean(self.frames_mfcc_buffer, axis=0)
        mfcc_std = np.std(self.frames_mfcc_buffer, axis=0)

        return np.divide(np.subtract(mfcc, mfcc_mean), mfcc_std)

    def __update_noise_buffer(self, frame):
        """
        Save new unvoiced frame
        """
        noise_spec_mag = get_spec_mag(frame, self.fft_n)

        if len(self.noise_buffer) == SKLearnAnalyzer.NOISE_BUFFER_SIZE:
            self.noise_buffer.pop(0)

        self.noise_buffer.append(noise_spec_mag)

    def __update_frames_buffers(self, frame):
        spec_mag = get_spec_mag(frame, self.fft_n)
        subtracted_spec = self.__noise_spec_subtraction(spec_mag)
        frame_mfcc = get_mfcc_from_spec(spec_mag, self.filterbank, self.mfcc_num)

        if len(self.frames_buffer) == SKLearnAnalyzer.FRAMES_BUFFER_SIZE:
            self.frames_buffer.pop(0)
            self.frames_mfcc_buffer.pop(0)

        self.frames_buffer.append(frame)
        self.frames_mfcc_buffer.append(frame_mfcc)