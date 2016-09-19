from file_processing import create_file_gen, create_random_file_gen, features_number
import numpy as np
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

musan_unvoiced_path = cur_dir + '/musan/unvoiced.csv'
musan_music_path = cur_dir + '/musan/music.csv'
tedlium_speech_path = cur_dir + '/tedlium/voiced.csv'

musan_noise_features_num = features_number(musan_unvoiced_path)
musan_music_features_num = features_number(musan_music_path)
tedlium_speech_features_num = features_number(tedlium_speech_path)


MUSAN_MUSIC = 0
MUSAN_NOISE = 1
TEDLIUM_SPEECH = 2


def get_features_number(total_size=1.0, use_musan_noise=True,
                        use_musan_music=True, use_tedlium_speech=True):

    features_num =  []

    if use_musan_noise:
        features_num.append(int(total_size*musan_noise_features_num))

    if use_musan_music:
        features_num.append(int(total_size*musan_music_features_num))

    if use_tedlium_speech:
        features_num.append(int(total_size*tedlium_speech_features_num))

    return features_num


def random_features_generator(total_size=1.0,
                              use_musan_noise=True, use_musan_music=True,
                              use_tedlium_speech=True):

    features_num = []
    generators = []

    if use_musan_noise:
        features_num.append(int(total_size * musan_noise_features_num))
        generators.append(musan_random_noise_gen(features_num[-1]))

    if use_musan_music:
        features_num.append(int(total_size*musan_music_features_num))
        generators.append(musan_random_music_gen(features_num[-1]))

    if use_tedlium_speech:
        features_num.append(int(total_size * tedlium_speech_features_num))
        generators.append(tedlium_random_speech_gen(features_num[-1]))

    total_features = sum(features_num)

    features_counter = 0
    while features_counter < total_features:
        total_batch = sum(features_num) + 0.0
        w = [(db + 0.0) / total_batch for db in features_num]

        if total_batch % 10000 == 0:
            print("Remaining samples %d" % total_batch)

        sample_index = np.random.randint(len(features_num))
        mw = max(w)
        b = 0

        for i in range(len(features_num)):
            if features_counter == total_features:
                break

            b += np.random.random() * 2 * mw
            while w[sample_index] <= b:
                b -= w[sample_index]

                if sample_index == len(features_num) - 1:
                    sample_index = 0
                else:
                    sample_index += 1

            try:
                features, cls = generators[sample_index].next()
                features_num[sample_index] -= 1
                features_counter += 1

                yield features, cls

            except StopIteration:
                print("\nStop iteration exception:")
                print("\tSample index %d" % sample_index)
                print("\tGenerator: " + str(generators[sample_index]))

                raise StopIteration()


def musan_noise_gen(batch_size=0):
    return create_file_gen(musan_unvoiced_path, batch_size)


def musan_music_gen(batch_size=0):
    return create_file_gen(musan_music_path, batch_size)


def tedlium_speech_gen(batch_size=0):
    return create_file_gen(tedlium_speech_path, batch_size)


def musan_random_noise_gen(batch_size=0):
    return create_random_file_gen(musan_unvoiced_path, batch_size)


def musan_random_music_gen(batch_size=0):
    return create_random_file_gen(musan_music_path, batch_size)


def tedlium_random_speech_gen(batch_size=0):
    return create_random_file_gen(tedlium_speech_path, batch_size)
