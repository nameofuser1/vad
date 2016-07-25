from numpy import fft
import numpy as np

class Noise(object):

    def __init__(self, fft_n):
        self.noise_frames = []
        self.fft_n = fft_n
        self.spectral_mean = None
        self.n_noise_frames = 0
        self.frames_buffer = []


    def substraction(self, frame):
        return fft.fft(frame*np.hanning(self.fft_n), self.fft_n)[0:self.fft_n/2] - self.spectral_mean

    def calculate_spectral_mean(self):
        frames_for_mean = []
        for frame in self.frames:
            frames_for_mean.append(fft.fft(frame*np.hanning(self.fft_n), self.fft_n))
        return np.mean(frames_for_mean, axis=0)

    def load_noise_frames(self, noise_frames, n_noise_frames):
        for frame in noise_frames:
            self.noise_frames.append(frame)
        self.spectral_mean = self.calculate_spectral_mean()

    def ltsd(self):
        frames_buffer_after_fft = []
        for frames_buffer in self.frames_buffer:
            frames_buffer_after_fft.append(np.fft.fft(frames_buffer*np.hanning(self.fft_n)))

        ltsd = 0.0

        for i in range(self.fft_n):
            max_for_i = 0.0
            for frame in frames_buffer_after_fft:
                frame_mag = np.sqrt(np.real(frame[i]) ** 2 + np.imag(frame[i]) ** 2)
                if frame_mag > max_for_i:
                    max_for_i = frame_mag

            noise_mag = np.sqrt(np.real(self.spectral_mean[i]) ** 2 + np.imag(self.spectral_mean[i]) ** 2)
            ltsd += (max_for_i**2)/(noise_mag**2)
        return 10* np.log10(ltsd/(self.fft_n+0.0))
