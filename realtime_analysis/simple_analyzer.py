import numpy as np
from pyAudioAnalysis.audioFeatureExtraction import stEnergy, stZCR
import logging
from analyser import Analyser


class SimpleAnalyser(Analyser):

    TEMP_BUFFER_SIZE = 20
    TEMP_ACTIVE_THRESHOLD = 5
    TEMP_INACTIVE_THRESHOLD = 15

    def __init__(self, frame_rate, frame_size, noise_buf_len):
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.spectral_bands = 4
        self.spectral_bin_width = 0
        self.noise_buf_len = noise_buf_len

        self.silence = True
        self.temp_buffer = [False]*self.TEMP_BUFFER_SIZE

        handler = logging.FileHandler("realtime.log", "w")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

        frames_handler = logging.FileHandler("frames.log", "w")
        frames_handler.setLevel(logging.DEBUG)
        frames_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

        self.frames_logger = logging.getLogger("FRAMES")
        self.frames_logger.setLevel(logging.DEBUG)
        self.frames_logger.addHandler(frames_handler)

        self.frame_number = 0

        self.fft_extended_zeros = 0
        self.fftn = 0
        self.fftn_for_band = 0

        self.noise_pointer = 0
        self.noise_frames = np.zeros((noise_buf_len, frame_size), np.float64)

        self.energy_thresh = 0.0
        self.energy_std = 0.0

        self.energy_k = 2.0

        self.spectral_std_thresh = 0.0
        self.spectal_std_k = 2.0

        self.spectral_energy_bands_thresh = np.zeros((self.spectral_bands,), np.float64)
        self.spectral_energy_bands_k = 3.0

        self._choose_optimal_fft_size()
        self.initialized = False

        self.inactive_in_row = 5
        self.active_in_row = 0

    #
    #   Close logger files handlers
    #
    def de_init(self):
        for handler in self.logger.handlers:
            handler.close()

        for handler in self.frames_logger.handlers:
            handler.close()

    #
    #   Feed new frame to analyse
    #
    def feed_frame(self, frame):

        if not self.initialized:
            raise Exception("Analyser not initialized")

        if len(frame) != self.frame_size:
            raise Exception("Wrong frame size. Expected " + str(self.frame_size) + " bytes. Got " + str(len(frame)))

        self.frame_number += 1
        self.frames_logger.debug("Frame " + str(self.frame_number) + ":\r\n" + str(frame)+"\r\n")
        self.logger.debug("Frame " + str(self.frame_number))

        frame_status = self._classify_frame(frame)
        self._add_status_to_temp_buffer(frame_status)
        active, inactive = self._get_temp_buffer_statuses()

        if frame_status:
            if self.silence:
                if active >= self.TEMP_ACTIVE_THRESHOLD:
                    self.silence = False

            self.logger.debug("_ACTIVE\r\n")
            return True
        else:
            if self.silence:
                self._add_inactive_frame(frame)
                self.logger.debug("_INACTIVE\r\n")
                return False
            else:
                if inactive >= self.TEMP_INACTIVE_THRESHOLD:
                    self._add_inactive_frame(frame)
                    self.silence = True

                self.logger.debug("_ACTIVE\r\n")
                return True

    #
    #   Saves initial `noise_buf_len` frames
    #   Which are assumed to be noise
    #
    #   All initial thresholds as mean.
    #
    def load_init_inactive_frames(self, frames):
        if len(frames) != self.noise_buf_len:
            raise Exception("Expected " + str(self.noise_buf_len) + " initial frames. Got " + str(len(frames)))

        self.frames_logger.debug("Loading inactive frames")
        for i in range(self.noise_buf_len):
            np.put(self.noise_frames[i], range(self.frame_size), frames[i])
            self.frames_logger.debug("Noise frame " + str(i) + ":\r\n" + str(frames[i]) + "\r\n")

        self.energy_thresh = self._inactive_mean_st_energy()
        self.spectral_energy_bands_thresh = self._inactive_spectral_energy_mean_bands()
        self.spectral_std_thresh = self._inactive_mean_spectral_std()

        self.logger.debug("Init energy threshold: " + str(self.energy_thresh))
        self.logger.debug("Init spectral energy threshold " + str(self.spectral_energy_bands_thresh))
        self.logger.debug("Init spectral std threshold " +str(self.spectral_std_thresh))
        self.logger.debug("\r\n")

        self.initialized = True

    #
    #   Returns True if frame is ACTIVE.
    #   Otherwise returns False.
    #
    def _classify_frame(self, frame):

        if self._is_spectral_energy_active(frame):
            self.logger.debug("Passed multiband")

            if self._is_zrc_active(frame):
                self.logger.debug("Passed std")
                return True

        if self._is_zrc_active(frame):
            self.logger.debug("Passed zcr")

            if self._is_energy_active(frame):
                self.logger.debug("Passed energy")

                if self._is_spectral_std_active(frame):
                    self.logger.debug("Passed spectral std")

                    return True

        return False

    def _is_energy_active(self, frame):
        frame_energy = stEnergy(frame)

        self.logger.debug("Frame energy: " + str(frame_energy))
        self.logger.debug("Energy threshold " + str(self.energy_thresh))

        return stEnergy(frame) > self.energy_k * self.energy_thresh

    #
    #   Band 0-1kHz must be active and 2 of the others
    #
    def _is_spectral_energy_active(self, frame):

        bands = self._get_spectral_bands(frame)
        bands_mean = np.zeros((4,), np.float64)

        for i in range(self.spectral_bands):
            bands_mean[i] = stEnergy(bands[i])

        self.logger.debug("Spectral bands energies: " + str(bands_mean))
        self.logger.debug("Spectral bands thresholds " + str(self.spectral_energy_bands_thresh))

        if bands_mean[0] > self.spectral_energy_bands_thresh[0] * self.spectral_energy_bands_k:

            active_bands = 0
            for i in range(1, self.spectral_bands):
                if bands_mean[i] > self.spectral_energy_bands_thresh[i] * self.spectral_energy_bands_k:
                    active_bands += 1

            if active_bands >= 2:
                return True

        return False

    #
    #   i.e. spectral flatness
    #
    def _is_spectral_std_active(self, frame):
        frame_std = np.std(self._fft(frame))

        self.logger.debug("Spectral std " + str(frame_std))
        self.logger.debug("Spectral std threshold " + str(self.spectral_std_thresh))

        return frame_std > self.spectral_std_thresh * self.spectal_std_k

    def _is_zrc_active(self, frame):

        frame_zcr = stZCR(frame)*self.frame_size
        self.logger.debug("ZCR " + str(frame_zcr))

        return 20 >= frame_zcr >= 5

    #
    #   Choose optimal fft len
    #   And calculate spectral bin width
    #
    def _choose_optimal_fft_size(self):

        val = 2

        while val < self.frame_size:
            val <<= 1

        self.fftn = val
        self.fft_extended_zeros = (val - self.frame_size) / 2 + (val - self.frame_size) % 2
        self.spectral_bin_width = (self.frame_rate+0.0) / (self.fftn+0.0)
        self.fftn_for_band = int(1000 / self.spectral_bin_width)

        self.logger.debug("FFTN is set to " + str(self.fftn))
        self.logger.debug("Spectral band width is " + str(self.spectral_bin_width))
        self.logger.debug("FFTN for band is set to " + str(self.fftn_for_band))

    #
    #   If frame is inactive by classifier then call it on it
    #
    #   Update time-energy threshold
    #   Update spectral bands energies thresholds
    #   Update spectral flatness threshold
    #
    def _add_inactive_frame(self, frame):

        np.put(self.noise_frames[self.noise_pointer], range(self.frame_size), frame)

        self._update_energy_threshold()
        self._update_spectral_energy_bands_threshold()
        self._update_spectral_std_threshold()

        if self.noise_pointer == self.noise_buf_len - 1:
            self.noise_pointer = 0
        else:
            self.noise_pointer += 1

    #
    #   Update spectral standart deviation threshold
    #
    def _update_spectral_std_threshold(self):
        p = 0.25

        self.spectral_std_thresh = (1-p) * self.spectral_std_thresh + p * self._inactive_mean_spectral_std()

    #
    #   Return mean of spectral energies standard deviations of inactive frames
    #
    def _inactive_mean_spectral_std(self):
        return np.mean(self._inactive_spectral_std())

    #
    #   Return array of spectral standart deviations for inactive frames:
    #       [0] - frame 0 spectral std
    #       [1] - frame 1 spectral std
    #       ...
    #
    def _inactive_spectral_std(self):
        std = np.zeros((self.noise_buf_len,), np.float64)

        for i in range(self.noise_buf_len):
            fft = self._fft(self.noise_frames[i])
            std[i] = np.std(fft)

        return std

    #
    #   Update threshold for time-energy
    #   P NOW CONST
    #
    def _update_energy_threshold(self):
        p = 0.25
        self.energy_thresh = (1-p) * self.energy_thresh + p * self._inactive_mean_st_energy()

    #
    #   Returns mean of inactive frames energies:
    #
    def _inactive_mean_st_energy(self):
        return np.mean(self._inactive_st_energies())

    #
    #   Returns numpy array of energies of inactive frames in noise buffer:
    #       [0] --- energy of frame 0
    #       [1] --- energy of frame 1
    #       ...
    #
    def _inactive_st_energies(self):

        energy = np.zeros((self.noise_buf_len,), np.uint32)

        for i in range(self.noise_buf_len):
            energy[i] = stEnergy(self.noise_frames[i])

        return energy

    #
    #   Updates thresholds for each spectral band
    #   P NOW CONST
    #
    def _update_spectral_energy_bands_threshold(self):
        # now const
        p = 0.25

        new_noise_energies = self._inactive_spectral_energy_mean_bands()
        for i in range(self.spectral_bands):
            self.spectral_energy_bands_thresh[i] = (1-p)*(self.spectral_energy_bands_thresh[i]) + \
                                                   p*new_noise_energies[i]

    #
    #   Return numpy array:
    #       [0] --- mean energy for band 0-1kHz
    #       [1] --- mean energy for band 1-2kHz
    #       [2] --- mean energy for band 2-3kHz
    #       [3] --- mean energy for band 3-4kHz
    #       ...
    #
    def _inactive_spectral_energy_mean_bands(self):
        return np.mean(self._inactive_spectral_energy_bands(), axis=0)

    #
    #   Return numpy array of energies of spectral bands:
    #       [i][0] --- 0-1kHz
    #       [i][1] --- 1-2kHz
    #       [i][2] --- 2-3kHz
    #       [i][3] --- 3-4kHz
    #       ...
    #
    def _inactive_spectral_energy_bands(self):

        bands_energies = np.zeros((self.noise_buf_len, self.spectral_bands), np.float64)

        for i in range(self.noise_buf_len):
            frame = self.noise_frames[i]
            frame_bands = self._get_spectral_bands(frame)

            for j in range(self.spectral_bands):
                bands_energies[i][j] = stEnergy(frame_bands[j])

        return bands_energies

    #
    #   Return spectral bands:
    #       [0][0-1000] --- 0-1kHz
    #       [1][0-1000] --- 1-2kHz
    #       [2][0-1000] --- 2-3kHz
    #       [3][0-1000] --- 3-4kHz
    #       ...
    #
    def _get_spectral_bands(self, frame):

        if self.fftn < self.spectral_bands*self.fftn_for_band:
            raise Exception("Can't get spectral bands. FFTN is small")

        bands = np.zeros((self.spectral_bands, self.fftn_for_band), np.float64)
        fft = self._fft(frame)

        for i in range(self.spectral_bands):
            np.put(bands[i], range(self.fftn_for_band), fft[i*self.fftn_for_band:(i+1)*self.fftn_for_band])

        return bands

    #
    #   Compute short-time fft
    #
    def _fft(self, frame):

        if self.fft_extended_zeros > 0:
            frame_to_trans = np.concatenate([np.zeros((self.fft_extended_zeros,), np.float64),
                                             frame,
                                             np.zeros((self.fft_extended_zeros,), np.float64)])
        else:
            frame_to_trans = frame[0:self.fftn]

        fft = np.fft.fft(frame_to_trans)
        fft = np.sqrt(np.real(fft) ** 2 + np.imag(fft) ** 2)

        return fft

    #
    #   Algorithm advises that for updating thresholds
    #
    @staticmethod
    def get_new_p(sigma_new, sigma_old):
        y = sigma_new / sigma_old

        if y >= 1.25:
            return 0.25

        if 1.25 >= y >= 1.10:
            return 0.20

        if 1.10 >= y >= 1.0:
            return 0.15

        if 1.0 >= y:
            return 0.10

    #
    #   Return (number of active frames, number of inative frames)  in temp buffer
    #
    def _get_temp_buffer_statuses(self):
        active = 0
        inactive = 0

        self.logger.debug(str(self.temp_buffer))

        for status in self.temp_buffer:
            if status:
                active += 1
            else:
                inactive += 1

        return active, inactive

    #
    #   Add frame status to circular buffer
    #
    def _add_status_to_temp_buffer(self, status):

        if len(self.temp_buffer) == self.TEMP_BUFFER_SIZE:
            self.temp_buffer.pop(0)
        self.temp_buffer.append(status)











