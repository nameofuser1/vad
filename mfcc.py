import numpy as np


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


def make_mel_bins(sample_rate, hzs, fft_n):
    mel_bin_numbers = []

    for hz in hzs:
        mel_bin_numbers.append(np.floor((fft_n+1) * hz / sample_rate))

    return mel_bin_numbers


def get_mel_filterbanks(low_hz, up_hz, fft_n, n_filters, sample_rate):

    hzs = hz_from_mel(mel_from_hz(low_hz, up_hz, n_filters))

    print('Len of hz points ' + str(len(hzs)))

    # Convert hz to FFT bin number
    mels_bin = make_mel_bins(sample_rate, hzs, fft_n)

    K = np.arange(0, fft_n/2, 1)
    filters = []

    for m in range(1, n_filters+1):
        filterbank = np.zeros(fft_n/2)

        for k in K:
            if (k >= mels_bin[m-1]) and (k <= mels_bin[m]):
                filterbank[k] = (k - mels_bin[m-1] + 0.0) / (mels_bin[m] - mels_bin[m-1] + 0.0)

            elif (k >= mels_bin[m]) and (k <= mels_bin[m+1]):
                filterbank[k] = -(k - mels_bin[m+1] + 0.0) / (mels_bin[m+1] - mels_bin[m] + 0.0)
        filters.append(filterbank)

    return filters


def get_mfcc(fft_n, frame, n_filters, low_hz, up_hz, sample_rate):
    frame_after_fft = np.absolute(np.fft.fft(frame, fft_n)[0:fft_n/2]/fft_n)
    filters = get_mel_filterbanks(low_hz, up_hz, fft_n, n_filters, sample_rate)
    coefs = []

    for filter in filters:
        coefs.append(np.log(np.dot(filter, frame_after_fft)))

    frames_cepstral = []

    for k in range(n_filters):
        coef = 0

        for j in range(n_filters):
            coef += coefs[k]*np.cos(k*(2*j-1)*np.pi / 2*n_filters)

        frames_cepstral.append(round(coef,2))

    return np.around(np.array(frames_cepstral), decimals = 2)


def get_deltas(mfcc2, mfcc1):
    return np.subtract(mfcc2, mfcc1)
