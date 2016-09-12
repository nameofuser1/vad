import sounddevice as sd
import wave
import numpy as np
from realtime_analysis.sklearn_analyser import SKLearnAnalyzer
from recoder import Recoder




if __name__ == '__main__':

    frame_rate = 16000
    period = 0.025                         # ms
    frame_size = int(period * frame_rate)
    noise_frames_num = 5
    channels = 1
    dtype = 'int16'

    analyser = SKLearnAnalyzer('./learning/classifiers/decision_classifier.cls', fft_n=512)
    active_frames_counter = 0
    active_frames = b''
    noise_frames = []

    frames_counter = 0
    init_passing = 0
    voiced_frames = []
    voice_start = -1

    rec = Recoder(frame_rate, period, channels, dtype)
    rec.start()

    try:
        while len(active_frames) < 16000*5:

            while rec.empty():
                pass

            np_frame = rec.read()

            # passing first 10 frames because somehow they are really noisy
            if init_passing < 10:
                init_passing += 1
                continue

            # filling in frames buffer first time
            if len(noise_frames) < noise_frames_num:
                noise_frames.append(np_frame)

                if len(noise_frames) == noise_frames_num:
                    analyser.load_init_inactive_frames(noise_frames)

            else:
                frames_counter += 1

                active_frame = analyser.feed_frame(np_frame)
                if active_frame is not None:
                    print("Got active frame")
                    active_frames_counter += 1
                    active_frames += np_frame.tobytes()

    finally:

        print("Got active frames " + str(active_frames_counter))
        print("Voiced frames " + str(voiced_frames))

        wf = wave.open("pyaudio.wav", "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(frame_rate)
        wf.writeframesraw(active_frames)
        wf.close()

    exit(1)





