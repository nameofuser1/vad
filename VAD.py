
# coding: utf-8

# In[54]:

import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pylab as plt



# In[3]:

#file = wavfile.read("C:\\Users\\User\Desktop\VAD\knocking.wav", False)[1]

# In[5]:
def get_file_len(file):
    raw_file = wavfile.read(file, False)[1]
    b = np.zeros(2*len(raw_file), np.int8)
    k = 0
    while k<len(raw_file):
        b[2*k] = (raw_file[k]>>8) & 0xFF
        b[2*k+1] % 0xFF
        k+=1
    return len(b)

#%%
def get_frame_from_file(frame_size,first_frame, file):
    raw_file = wavfile.read(file, False)[1]
    b = np.zeros(2*len(raw_file), np.int8)
    k = 0
    while k<len(raw_file):
        b[2*k] = (raw_file[k]>>8) & 0xFF
        b[2*k+1] % 0xFF
        k+=1
    return b[first_frame: first_frame+frame_size]
# In[6]:
frames =get_frame_from_file(400,0,"C:\\Users\\User\Desktop\VAD\knocking.wav")
# In[29]:
#i = 0
#zeros = np.zeros(48)
#frames=[]
#while i<1600:
#    frames.append(np.fft.fft(np.concatenate((zeros,b[i:i+160],zeros)))[0:128])
#    i=i+160
# In[32]:
#
#print(len(frames[0]))
#noise = np.mean(frames, axis=0)
# In[33]:
#i =1600
#frames_clean = []
#while i <len(b):
#    frames_clean.append(np.fft.fft(np.concatenate((zeros,b[i:i+160],zeros)))[0:128] - noise)
#    i+=160


# In[34]:
#
#res = np.array([], dtype = np.int8)
#

# In[35]:

#for frame in frames_clean:
#    res = np.concatenate((res, frame))
#

# In[46]:

#result = wavfile.write("result", 16000, res)


# In[85]:

class noise(object):
    from numpy import fft
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
            frames_for_mean.append(fft.fft(frames*np.hanning(self.fft_n), self.fft_n))
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


# In[81]:
#
#x = np.linspace(-np.pi, np.pi, 512)
##plt.plot(x, np.sin(50*x))
#
#fft_1 = np.fft.fft((np.sin(100*x)+np.sin(50*x))*np.hanning(512))


# In[68]:

def plot_fft(f):
    N = f.shape[0]
    mag = (np.sqrt(np.real(f) ** 2 + np.imag(f) ** 2)) / N
    #freq = np.fft.fftfreq(N, step)
    freq = np.concatenate([np.arange(0, N/2), np.arange(-N/2, 0)])

    plt.plot(freq, mag)
    plt.show()

    return freq, mag


# In[82]:

#plot_fft(fft_1)


# In[ ]:

def Mel_from_Hz(first_hz, upper_Hz, n_bins):
    mels = []
    first_mel = 1125.0*np.log(1.0+first_hz/700.0)
    #mels.append(first_mel)
    last_mel = 1125.0*np.log(1.0+upper_Hz/700.0)
    delta  = (last_mel - first_mel)/(n_bins+1)
    for i in range(n_bins+1):
        mels.append(first_mel + i*delta)
    mels.append(last_mel)
    mels.sort()
    return mels
# In[ ]:
def one_Hz_from_Mel(mel):
    return 700*(np.exp(mel/1125)-1)
# In[ ]:
def all_Hz_from_Mel(mels):
    mels_Hz = list(map(one_Hz_from_Mel,mels))
    return mels_Hz
# In[ ]:
Hzs= all_Hz_from_Mel(Mel_from_Hz(300, 8000,10))
# In[ ]:

#f(i) = floor((nfft+1)*h(i)/upper_frequency)  #upper_frequency is a sample rate
def make_mel_bins(upper_frequency, Hzs, fft_n):
    mel_bin_numbers = []
    for Hz in Hzs:
        mel_bin_numbers.append(np.floor((fft_n+1)*Hz/upper_frequency))
    return mel_bin_numbers

# In[ ]:
mels_bin = make_mel_bins(16000, Hzs, 512)

# In[ ]:

K = np.arange(0, 256, 1)
filters = []
x_plot = []
for m in range(1, 11):
    filterbank = np.zeros(256)
    
    for k in K:
        if (k >= mels_bin[m-1]) and  (k<= mels_bin[m]): 
            filterbank[k]  = (k-mels_bin[m-1]+0.0)/(mels_bin[m]-mels_bin[m-1]+0.0)
        elif (k >= mels_bin[m]) and  (k<= mels_bin[m+1]):
            filterbank[k]  = -(k-mels_bin[m+1]+0.0)/(mels_bin[m+1]-mels_bin[m]+0.0)
    filters.append(filterbank)
    
X = list(K)
bin_to_Hz = lambda x: x*8000.0/256.0 
bins_as_Hz=list(map(bin_to_Hz, X))

for filter in filters:
    plt.plot(bins_as_Hz, filter, 'r-')

#%%

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
        
    X = list(K)
    bin_to_Hz = lambda x: x*sample_rate*1.0/(fft_n*1.0)
    bins_as_Hz=list(map(bin_to_Hz, X))
    return filters
    #for filter in filters:
    #    plt.plot(bins_as_Hz, filter, 'r-')
    
#%%
get_mel_filterbanks(300, 8000, 256, 10, 16000)
#%%

def get_spectral_frames(fft_n, frame, n_filters, low_Hz, up_Hz, sample_rate ):
    frame_after_fft = np.absolute(np.fft.fft(frame, fft_n)[0:fft_n/2]/fft_n)
    filters = get_mel_filterbanks(low_Hz, up_Hz, fft_n, n_filters, sample_rate)
    coefs = []
    for filter in filters:
        coefs.append(np.log(np.dot(filter,frame_after_fft)))
    frames_cepstral = []
    for k in range(n_filters):
        coef = 0
        for j in range(n_filters):
            coef +=coefs[k]*np.cos(k*(2*j-1)*np.pi / 2*n_filters)
        frames_cepstral.append(round(coef,2))
        
    return np.around(np.array(frames_cepstral), decimals = 2)
#%%
def get_deltas(fft_n, n_filters, low_Hz, up_Hz, sample_rate, file, step, size):
    frames = get_spectral_of_file(fft_n, n_filters, low_Hz, up_Hz, sample_rate, file,step,size)
    n = len(frames)
    deltas = [[""]*n_filters]*n
    for k in range(n_filters): 
        for i in range(n):
            if i == 0:
                deltas[0][k]=frames[1][k]-frames[0][k]
            elif i == n-1:
                deltas[n-1][k]=frames[n-1][k]- frames[n-2][k]
            else:
                deltas[i][k]=frames[i+1][k]-frames[i-1][k]
    return deltas

#%%

def get_all(file_way,length_samples, step):
    import csv
    k = np.mod((get_file_len(file_way) - 400), 160)
    samples = np.array([])
    deltas = np.array([])
    for i in range(k):
        np.append(samples, np.array(get_frame_from_file(length_samples,step*i, file_way)))
        np.append(deltas, np.array(get_deltas(get_frame_from_file(length_samples,step*i, file_way))))
        i+=1
    result = [np.concatenate(x) for x in zip(samples,deltas)]
    csv.writer(open('blank.csv', 'w', newline=''), delimiter=',').writerows(result)
#%%
def get_samples_from_file(file):  #это новая функция, получает просто семплы из файла
    raw_file = wavfile.read(file, False)[1]
    b = np.zeros(2*len(raw_file), np.int8)
    k = 0
    while k<len(raw_file):
        b[2*k] = (raw_file[k]>>8) & 0xFF
        b[2*k+1] % 0xFF
        k+=1
    return b
#%%
def get_spectral_of_file(fft_n, n_filters, low_Hz, up_Hz, sample_rate, file, step, size): # слайсами делит семплы и выдает массив в котором каждый элемент это вектор спектрала и дельты
    spectrals = []
    deltas = []
    samples = np.array(get_samples_from_file(file))
    k = np.floor_divide((len(samples) - size), step)
    result = []
    for i in range(k):
        empty = []
        spectrals.append(get_spectral_frames(fft_n, samples[i*step:i*step+size], 
                                                 n_filters, low_Hz, up_Hz, sample_rate))
        empty.append(get_spectral_frames(fft_n, samples[i*step:i*step+size], 
                                                 n_filters, low_Hz, up_Hz, sample_rate))                                         
        #deltas.append(get_deltas(fft_n, samples[i*step:i*step+size], n_filters, low_Hz, up_Hz, sample_rate))
        #empty.append(get_deltas(fft_n, samples[i*step:i*step+size], n_filters, low_Hz, up_Hz, sample_rate))
        #result.append(empty)
    #result = [np.concatenate(x) for x in zip(spectrals,deltas)]
    return spectrals

#%%
def get_spectral_and_deltas_of_file(fft_n, n_filters, low_Hz, up_Hz, sample_rate, file, step, size): # слайсами делит семплы и выдает массив в котором каждый элемент это вектор спектрала и дельты
    spectrals = []
    deltas = []
    samples = np.array(get_samples_from_file(file))
    k = np.floor_divide((len(samples) - size), step)
    result = []
    for i in range(k):
        empty = []
        spectrals.append(get_spectral_frames(fft_n, samples[i*step:i*step+size], 
                                                 n_filters, low_Hz, up_Hz, sample_rate))
        empty.append(get_spectral_frames(fft_n, samples[i*step:i*step+size], 
                                                 n_filters, low_Hz, up_Hz, sample_rate))                                         
        deltas.append(get_deltas(fft_n, samples[i*step:i*step+size], n_filters, low_Hz, up_Hz, sample_rate, file, step,size fft_n, n_filters, low_Hz, up_Hz, sample_rate, file, step, siz))
        empty.append(get_deltas(fft_n, samples[i*step:i*step+size], n_filters, low_Hz, up_Hz, sample_rate, file, step, size))
        result.append(empty)
    result = [np.concatenate(x) for x in zip(spectrals,deltas)]
    return result


#%%

def write_csv(directory,fft_n, n_filters, low_Hz, up_Hz, sample_rate, file, step, size):
    import csv
    import os
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            result = get_spectral_of_file(fft_n, n_filters, low_Hz, up_Hz, sample_rate, file, step, size)
            csv.writer(open('blank.csv', 'w', newline=''), delimiter=',').writerows(result)
    
#%%
len(get_spectral_of_file(512, 25, 300, 8000, 16000, "C:\\Users\\User\Desktop\VAD\music.wav", 160, 400))
#%%
get_spectral_frames(512, frame, n_filters, low_Hz, up_Hz, sample_rate )
#%%
len(get_deltas(512, 25, 300, 8000, 16000, "C:\\Users\\User\Desktop\VAD\music.wav", 160, 400))