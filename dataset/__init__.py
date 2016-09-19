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


def get_features_number(test_train_ratio=0.75, use_musan_noise=True,
                        use_musan_music=True, use_tedlium_speech=True):
    features_num = []

    if use_musan_noise:
        features_num.append(int(musan_noise_features_num*test_train_ratio))

    if use_musan_music:
        features_num.append(int(musan_music_features_num*test_train_ratio))

    if use_tedlium_speech:
        features_num.append(int(tedlium_speech_feature_num*test_train_ratio))

    return features_num


def random_features_generator(test_train_ratio=0.75, batch_size=32,
                              use_musan_noise=True, use_musan_music=True,
                              use_tedlium_speech=True):
    batches = []
    generators = []

    if use_musan_noise:
        batches.append(int(musan_noise_features_num*test_train_ratio))
        generators.append(musan_random_noise_gen(batches[-1]))

    if use_musan_music:
        batches.append(int(musan_music_features_num*test_train_ratio))
        generators.append((musan_random_music_gen(batches[-1])))

    if use_tedlium_speech:
        batches.append(int(tedlium_speech_features_num*test_train_ratio))
        generators.append(tedlium_random_speech_gen(batches[-1]))

    features_counter = 0
    total_features = sum(batches) + 0.0
    print("Total features to load %d" % total_features)

    batch_x = []
    batch_y = []

    while features_counter < total_features:
        total_batch = sum(batches) + 0.0
        w = [(db + 0.0) / total_batch for db in batches]

        if total_batch % 10000 == 0:
            print("Remaining samples %d" % total_batch)

        sample_index = np.random.randint(len(batches))
        mw = max(w)
        b = 0

        for i in range(len(batches)):
            if features_counter == total_features:
                break

            b += np.random.random() * 2 * mw
            while w[sample_index] <= b:
                b -= w[sample_index]

                if sample_index == len(batches) - 1:
                    sample_index = 0
                else:
                    sample_index += 1

            try:
                features, cls = generators[sample_index].next()
                batches[sample_index] -= 1
                features_counter += 1

                batch_x.append(features)
                batch_y.append(cls)

                if len(batch_x) == batch_size:
                    yield batch_x, batch_y

            except StopIteration:
                print("\nStop iteration exception:")
                print("\tSample index %d" % sample_index)
                print("\tBatch size: %d" % batches[sample_index])
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
