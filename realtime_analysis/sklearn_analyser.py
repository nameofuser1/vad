import cPickle
from analyser import Analyser
from mfcc import get_mel_filterbanks, get_mfcc, get_deltas


class SKLearnAnalyzer(Analyser):

    BUFFER_SIZE = 5
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

        with open(fname) as f:
            self.classifier = cPickle.load(f)

    def load_init_inactive_frames(self, frames):
        if len(frames) != SKLearnAnalyzer.BUFFER_SIZE:
            raise ValueError("Number of inactive frame must be the same as BUFFER SIZE")

        self.frames_buffer = [get_mfcc(frame, self.fft_n, self.filterbank, self.mfcc_num) for frame in frames]

    def feed_frame(self, frame):
        processing_frame = self.frames_buffer[self.PROCESSING_FRAME_INDEX]
        processing_frame_mfcc = processing_frame

        prev_frame_mfcc = self.frames_buffer[self.PROCESSING_FRAME_INDEX - 1]
        prev_frame_first_deltas = get_deltas(processing_frame_mfcc, self.frames_buffer[self.processing_frame_index - 2])

        next_frame_mfcc = self.frames_buffer[self.processing_frame_index + 1]
        next_frame_first_deltas = get_deltas(self.frames_buffer[self.processing_frame_index + 2], processing_frame_mfcc)

        processing_frame_first_deltas = get_deltas(next_frame_mfcc, prev_frame_mfcc)
        processing_frame_second_deltas = get_deltas(next_frame_first_deltas, prev_frame_first_deltas)

        # Update circular buffer
        self.frames_buffer.pop(0)
        self.frames_buffer.append(get_mfcc(frame, self.fft_n, self.mel_filterbank, self.mfcc_num))

        features = []
        features.extend(processing_frame_mfcc)
        features.extend(processing_frame_first_deltas)
        features.extend(processing_frame_second_deltas)

        cls = self.classifier.predict(features)

        if cls == 1:
            return processing_frame
        elif cls == 0:
            return None
        else:
            raise AssertionError('Wrong classifier class')
