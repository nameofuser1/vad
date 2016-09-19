import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

from dataset import random_features_generator, get_features_number

EPOCHS_NUM = 25
BATCH_SIZE = 32

TRAIN_TEST_RATIO = 0.75
BATCH_SIZE = 32
TOTAL_SIZE = 0.0001
USE_MUSAN_NOISE = True
USE_MUSAN_MUSIC = True
USE_TEDLIUM_SPEECH = True


def get_batches_num(batch_size=32, total_size=1.0, use_musan_noise=True,
                    use_musan_music=True, use_tedlium_speech=True):

    f_num = get_features_number(total_size, use_musan_noise, use_musan_music, use_tedlium_speech)
    b_num = [n / batch_size for n in f_num]

    return b_num


def get_batches_offsets(test_train_ratio=0.75, batch_size=32, total_size=1.0, use_musan_noise=True,
                    use_musan_music=True, use_tedlium_speech=True):

    train_n, test_n = get_features_number(test_train_ratio, total_size, use_musan_noise, use_musan_music,
                                          use_tedlium_speech)

    train_bo = train_n % batch_size
    test_bo = test_n % batch_size

    return train_bo, test_bo


def get_batch(batch_size, features_gen):
    batch_x = []
    batch_y = []

    for features, cls in features_gen:
        while len(batch_x) < batch_size:
            batch_x.append(features)
            batch_y.append(cls)

        batch_y = np_utils.to_categorical(batch_y)

        return batch_x, batch_y


def batch_generator(epochs_num, batch_size=32, total_size=1.0, use_musan_noise=True,
                    use_musan_music=True, use_tedlium_speech=True):

    batch_n = sum(get_batches_num(batch_size, total_size, use_musan_noise, use_musan_music,
                              use_tedlium_speech))

    print("Total batches: %d" % batch_n)

    epoch = 0
    while epoch < epochs_num:
        features_gen = random_features_generator(total_size, use_musan_noise, use_musan_music,
                                                 use_tedlium_speech)
        train_counter = 0
        while train_counter < batch_n:
            batch_x, batch_y = get_batch(batch_size, features_gen)
            train_counter += 1

            yield batch_x, batch_y

        epoch += 1
        features_gen.close()


def get_train_test_generators(epochs_num, train_test_ratio=0.75, batch_size=32, total_size=1.0, use_musan_noise=True,
                                  use_musan_music=True, use_tedlium_speech=True):

    total_train_size = total_size*train_test_ratio
    total_test_size = total_size*(1.0-train_test_ratio)

    train_bn = get_batches_num(batch_size, total_train_size, use_musan_noise, use_musan_music, use_tedlium_speech)
    test_bn = get_batches_num(batch_size, total_test_size, use_musan_noise, use_musan_music, use_tedlium_speech)

    train_generator = batch_generator(epochs_num, batch_size, total_train_size, use_musan_noise, use_musan_music,
                                      use_tedlium_speech)

    test_generator = batch_generator(epochs_num, batch_size, total_test_size, use_musan_noise, use_musan_music,
                                     use_tedlium_speech)

    return (train_generator, sum(train_bn)), (test_generator, sum(test_bn))


if __name__ == "__main__":
    input_shape = (39,)

    model = Sequential()

    model.add(Dense(39, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adadelta')

    train_info, val_info = get_train_test_generators(1, TRAIN_TEST_RATIO, BATCH_SIZE, TOTAL_SIZE,
                                                     USE_MUSAN_NOISE, USE_MUSAN_MUSIC, USE_TEDLIUM_SPEECH)

    train_gen = train_info[0]
    train_samples = train_info[1]

    val_gen = val_info[0]
    val_samples = val_info[1]

    print("Number of train batches: %d" % train_samples)
    print("Number of validation batches: %d" % val_samples)

    model.fit_generator(train_gen, train_samples, EPOCHS_NUM, validation_data=val_gen,
                        nb_val_samples=val_samples, nb_worker=4)

    score = model.evaluate_generator(val_gen, val_samples, nb_worker=4)

    print("Validation score: " + str(score[0]))
    print("Validation accuracy: " + str(score[1]))





