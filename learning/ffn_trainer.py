import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

from dataset import random_features_generator, get_features_number

import numpy as np

TOTAL_SIZE = 0.01
TRAIN_TEST_RATIO = 0.75
BATCH_SIZE = 32
EPOCHS_NUM = 25

USE_MUSAN_NOISE = True
USE_MUSAN_MUSIC = True
USE_TEDLIUM_SPEECH = True

ACCURACY_THRESHOLD = 0.90

MODEL_NAME = 'first_model'
PREFIX = "./classifiers"


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
    batch_x = np.zeros((batch_size, 39), dtype=np.float32)
    batch_y = np.zeros(batch_size, dtype=np.uint8)

    for i in range(batch_size):
        features, cls = features_gen.next()
        np.put(batch_x[i], np.arange(39), features)
        batch_y[i] = cls

    batch_y = np_utils.to_categorical(batch_y, nb_classes=3)

    return batch_x, batch_y


def batch_generator(epochs_num, batch_size=32, total_size=1.0, use_musan_noise=True,
                    use_musan_music=True, use_tedlium_speech=True):

    batch_n = sum(get_batches_num(batch_size, total_size, use_musan_noise, use_musan_music,
                              use_tedlium_speech))

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
    model_ready = False
    input_shape = (39,)

    model = Sequential()

    model.add(Dense(64, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adadelta')

    print(model.summary())

    train_info, val_info = get_train_test_generators(EPOCHS_NUM, TRAIN_TEST_RATIO, BATCH_SIZE, TOTAL_SIZE,
                                                     USE_MUSAN_NOISE, USE_MUSAN_MUSIC, USE_TEDLIUM_SPEECH)

    epoch_counter = 0
    while epoch_counter < EPOCHS_NUM:
        print("Epoch %d" % epoch_counter)

        train_gen = train_info[0]
        train_samples = train_info[1]
        print("Batches per train epoch: " + str(train_samples))

        val_gen = val_info[0]
        val_samples = val_info[1]
        print("Batches for validation: " + str(val_samples))

        for i in range(train_samples):
            if epoch_counter > 15:
                print("Reading next")

            batch_x, batch_y = train_gen.next()

            if epoch_counter > 15:
                print("Training")

            model.train_on_batch(batch_x, batch_y)

            if (epoch_counter > 15) and (i % 100 == 0):
                print("Processed %d batches" % i)

        print("Validating...")
        val_score = model.evaluate_generator(val_gen, val_samples)

        print("Validation score: " + str(val_score[0]))
        print("Validation accuracy: " + str(val_score[1]))

        if (val_score[1] > ACCURACY_THRESHOLD) and not model_ready:
            print("Needed accuracy is achieved on epoch %d" % epoch_counter)
            print("Saving model...")

            with open(PREFIX+'/'+MODEL_NAME+'_achieved_accuracy.json', 'w') as model_f:
                model_f.write(model.to_json())

            model.save_weights(PREFIX+'/'+MODEL_NAME+'_achieved_accuracy.weights')

            model_ready = True

        epoch_counter += 1

    with open(PREFIX + '/' + MODEL_NAME + '_full_training.json', 'w') as model_f:
        model_f.write(model.to_json())

    model.save_weights(PREFIX + '/' + MODEL_NAME + '_full_training.weights')






