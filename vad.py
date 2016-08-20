import pyaudio
import wave

import numpy as np

from realtime_analysis.sklearn_analyser import SKLearnAnalyzer
from recoder import Recoder

if __name__ == '__main__':

    frame_rate = 16000
    frame_size = int(0.010 * 16000)  # 10 ms frame
    noise_frames_num = 5

    analyser = SKLearnAnalyzer('./classifiers/decision_classifier.cls')
    active_frames_counter = 0
    active_frames = b''
    noise_frames = []

    rec = Recoder(frame_rate=16000, period=160)
    rec.start()

    frames_counter = 0
    init_passing = 0
    voiced_frames = []
    voice_start = -1

    try:
        while len(active_frames) < 16000*5:

            if not rec.empty():
                raw_data = rec.read()

                # passing first 10 frames because somehow they are really noisy
                if init_passing < 10:
                    init_passing += 1
                    continue

                for data in raw_data:
                    frame = data[1]
                    np_frame = np.float64(np.fromstring(frame, np.int8))
                    # have to check it
                    np_frame_substracted = np_frame - np.sign(np_frame)*127

                    # filling in frames buffer first time
                    if len(noise_frames) < noise_frames_num:
                        noise_frames.append(np_frame_substracted)

                        if len(noise_frames) == noise_frames_num:
                            analyser.load_init_inactive_frames(noise_frames)

                    else:
                        frames_counter += 1

                        if analyser.feed_frame(np_frame_substracted):
                            active_frames_counter += 1
                            active_frames += frame

    finally:

        print("Got active frames " + str(active_frames_counter))
        print("Voiced frames " + str(voiced_frames))

        rec.stop()

        wf = wave.open("pyaudio.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt8))
        wf.setframerate(16000)
        wf.writeframesraw(active_frames)
        wf.close()



    exit(1)





