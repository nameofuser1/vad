# datasets
MUSAN_DATASET = 'musan'
TEDLIUM_DATASET = 'tedlium'

MUSAN_PATH = '/Users/bigmac/Documents/speech/musan'
MUSAN_SPEECH_PATH = None # [MUSAN_PATH + '/speech/us-gov', MUSAN_PATH + '/noise/free-sound']
MUSAN_TRANSCRIPTION_PATH = None
MUSAN_NOISE_PATH = [MUSAN_PATH + '/noise/sound-bible', MUSAN_PATH + '/noise/free-sound']
MUSAN_MUSIC_PATH = [MUSAN_PATH + '/music/fma', MUSAN_PATH + '/music/rfm', MUSAN_PATH + '/music/fma-western-art',
                    MUSAN_PATH + '/music/jamendo']

TEDLIUM_PATH = '/Users/bigmac/Documents/speech/tedlium'
TEDLIUM_SPEECH_PATH = [TEDLIUM_PATH + '/train/sph']
TEDLIUM_TRANSCRIPTION_PATH = TEDLIUM_PATH + '/train/stm'
TEDLIUM_NOISE_PATH = None
TEDLIUM_MUSIC_PATH = None


# MFCC related variables
SAMPLERATE = 16000
FRAME_SIZE = 400
FRAME_STEP = 160
LOW_HZ = 300
HIGH_HZ = 8000
FILTERBANKS_NUM = 26
MFCC_NUM = 13
FFT_N = 512

# Processing related variables
PROCESSES_NUM = 4

FILES_PER_STEP = 30

MAX_SPEECH_FILES = 500
MAX_NOISE_FILES = 2500
MAX_MUSIC_FILES = 2000

# Training variables
VOICED_FEATURES_NUM = 4500000
UNVOICED_FEATURES_NUM = 2000000

TRAIN_TEST_RATIO = 0.75
TRAIN_VALIDATION_RATIO = 0.75

NONE_VOICED = 0
VOICED = 1
MUSIC = 2