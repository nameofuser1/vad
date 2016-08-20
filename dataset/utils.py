import numpy as np


# For you Edward
def scale_features(features):
    scaled_features = features #unshared_copy(features)
    mfcc_buffer = []
    delta1_buffer = []
    delta2_buffer = []

    for file_features in scaled_features:
        for frame_features in file_features:
            mfcc_buffer.append(frame_features[0])
            delta1_buffer.append(frame_features[1])
            delta2_buffer.append(frame_features[2])

    mfcc_mean = np.mean(mfcc_buffer)
    delta1_mean = np.mean(delta1_buffer)
    delta2_mean = np.mean(delta2_buffer)

    mfcc_std = np.std(mfcc_buffer)
    delta1_std = np.std(delta1_buffer)
    delta2_std = np.std(delta2_buffer)

    for file_features in scaled_features:
        for frame_features in file_features:
            for i in range(len(frame_features[0])):
                frame_features[0][i] = (frame_features[0][i] - mfcc_mean) / mfcc_std
                frame_features[1][i] = (frame_features[1][i] - delta1_mean) / delta1_std
                frame_features[2][i] = (frame_features[2][i] - delta2_mean) / delta2_std

    return scaled_features


