import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

from dataset import random_features_generator, get_features_number

EPOCHS_NUM = 25
BATCH_SIZE = 32

TRAIN_RATIO = 0.01
VALIDATION_RATIO = TRAIN_RATIO = 0.3 * TRAIN_RATIO
TEST_RATION = 0.1 * TRAIN_RATION


def batch_generator(train_test_ratio=0.75, batch_size=32):
    features_gen = random_features_generator(train_test_ratio=train_test_ratio,
                                             batch_size=batch_size)

    try:
        for batch_x, batch_y in features_gen:
            batch_y = np_utils.to_categorical(batch_y)
            yield batch_x, batch_y

    except StopIteration():
        return  None, None


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

    total_train_features = get_features_number(train_test_ratio=TRAIN_RATIO)
    total_validation_features = get_features_number(train_test_ratio=VALIDATION_RATIO)
    total_test_features = get_features_number(train_test_ratio=TEST_RATIO)

    epoch = 0
    while epoch < EPOCHS_NUM:
        train_gen = batch_generator(TRAIN_RATIO)

        try:
            for ba

        except StopIteration():
            print("Epoch %d passed" % epoch)
            epoch += 1

            validation_gen = batch_generator(VALIDATION_RATIO)



