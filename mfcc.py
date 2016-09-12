import numpy as np
from scipy.fftpack import dct


def mel_from_hz(first_hz, upper_hz, n_bins):
    mels = []
    first_mel = 1125.0*np.log(1.0+first_hz/700.0)
    last_mel = 1125.0*np.log(1.0 + upper_hz / 700.0)

    delta = (last_mel - first_mel)/(n_bins+1)

    for i in range(n_bins+1):
        mels.append(first_mel + i*delta)

    mels.append(last_mel)
    mels.sort()

    return mels


def one_hz_from_mel(mel):
    return 700*(np.exp(mel/1125)-1)


def hz_from_mel(mels):
    mels_hz = list(map(one_hz_from_mel, mels))
    return mels_hz


def convert_to_fft_bins(sample_rate, hzs, fft_n):
    mel_bin_numbers = []

    for hz in hzs:
        mel_bin_numbers.append(np.floor((fft_n+1) * hz / sample_rate))

    return mel_bin_numbers


def get_mel_filterbanks(low_hz, up_hz, fft_n, n_filters, sample_rate):
    hzs = hz_from_mel(mel_from_hz(low_hz, up_hz, n_filters))

    # Convert hz to FFT bin number
    mels_bin = convert_to_fft_bins(sample_rate, hzs, fft_n)

    K = np.arange(0, fft_n/2, 1)
    filterbank = np.zeros((n_filters, fft_n/2))

    for m in range(1, n_filters+1):
        for k in K:
            if (k >= mels_bin[m-1]) and (k <= mels_bin[m]):
                filterbank[m-1, k] = (k - mels_bin[m-1] + 0.0) / (mels_bin[m] - mels_bin[m-1] + 0.0)

            elif (k >= mels_bin[m]) and (k <= mels_bin[m+1]):
                filterbank[m-1, k] = (mels_bin[m+1] - k + 0.0) / (mels_bin[m+1] - mels_bin[m] + 0.0)

    return filterbank


def get_spec_mag(frame, fft_n):
    frame = frame.astype(np.float32)
    return np.square(np.absolute(np.fft.fft(frame, fft_n)[0:fft_n / 2] / np.float32(fft_n)))


#
#   Number of MFCC is equal number of filters in filterbank
#
def get_mfcc(frame, fft_n, filterbank, mfcc_n):
    frame_after_fft = get_spec_mag(frame, fft_n)
    return get_mfcc_from_spec(frame_after_fft, filterbank, mfcc_n)


def get_mfcc_from_spec(spec, filterbank, mfcc_n):
    fbank_energies = np.dot(spec, filterbank.T)
    fbank_energies = np.where(fbank_energies==0, np.finfo(float).eps, fbank_energies)
    coefs = np.log10(fbank_energies)
    mfcc = dct(coefs, type=2, norm='ortho')[:mfcc_n]

    return lifter(mfcc)


def get_deltas(mfcc2, mfcc1):
    return np.subtract(mfcc2, mfcc1)


def lifter(cepstra, L=22):
    if L > 0:
        ncoeff = np.shape(cepstra)[0]
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra
