import numpy as np


def Mel_from_Hz(first_hz, upper_Hz, n_bins):
    mels = []
    first_mel = 1125.0*np.log(1.0+first_hz/700.0)
    last_mel = 1125.0*np.log(1.0+upper_Hz/700.0)

    delta  = (last_mel - first_mel)/(n_bins+1)

    for i in range(n_bins+1):
        mels.append(first_mel + i*delta)

    mels.append(last_mel)
    mels.sort()

    return mels


def one_Hz_from_Mel(mel):
    return 700*(np.exp(mel/1125)-1)


def all_Hz_from_Mel(mels):
    mels_Hz = list(map(one_Hz_from_Mel,mels))
    return mels_Hz

Hzs= all_Hz_from_Mel(Mel_from_Hz(300, 8000,10))


# upper_frequency is a sample rate
def make_mel_bins(upper_frequency, Hzs, fft_n):
    mel_bin_numbers = []
    for Hz in Hzs:
        mel_bin_numbers.append(np.floor((fft_n+1)*Hz/upper_frequency))
    return mel_bin_numbers


def get_mel_filterbanks(low_Hz, up_Hz, fft_n, n_filters, sample_rate):
    Hzs = all_Hz_from_Mel(Mel_from_Hz(low_Hz, up_Hz,n_filters))
    mels_bin = make_mel_bins(sample_rate, Hzs , fft_n)
    K = np.arange(0, fft_n/2, 1)
    filters = []

    for m in range(1, n_filters+1):
        filterbank = np.zeros(fft_n/2)

        for k in K:
            if (k >= mels_bin[m-1]) and  (k<= mels_bin[m]):
                filterbank[k]  = (k-mels_bin[m-1]+0.0)/(mels_bin[m]-mels_bin[m-1]+0.0)
            elif (k >= mels_bin[m]) and  (k<= mels_bin[m+1]):
                filterbank[k]  = -(k-mels_bin[m+1]+0.0)/(mels_bin[m+1]-mels_bin[m]+0.0)
        filters.append(filterbank)

    return filters


def get_mfcc(fft_n, frame, n_filters, low_Hz, up_Hz, sample_rate):
    frame_after_fft = np.absolute(np.fft.fft(frame, fft_n)[0:fft_n/2]/fft_n)
    filters = get_mel_filterbanks(low_Hz, up_Hz, fft_n, n_filters, sample_rate)
    coefs = []

    for filter in filters:
        coefs.append(np.log(np.dot(filter,frame_after_fft)))

    frames_cepstral = []

    for k in range(n_filters):
        coef = 0
        for j in range(n_filters):
            coef += coefs[k]*np.cos(k*(2*j-1)*np.pi / 2*n_filters)
        frames_cepstral.append(round(coef,2))

    return np.around(np.array(frames_cepstral), decimals = 2)


def get_deltas(mfcc2, mfcc1):
    return np.subtract(mfcc2, mfcc1)