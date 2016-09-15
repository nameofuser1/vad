from file_processing import create_file_gen
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

musan_unvoiced_path = cur_dir + '/musan/unvoiced.csv'
musan_music_path = cur_dir + '/musan/music.csv'
tedlium_speech_path = cur_dir + '/tedlium/voiced.csv'


def musan_noise_gen(batch_size):
    return create_file_gen(musan_unvoiced_path, batch_size)


def musan_music_gen(batch_size):
    return create_file_gen(musan_music_path, batch_size)


def tedlium_speech_gen(batch_size):
    return create_file_gen(tedlium_speech_path, batch_size)