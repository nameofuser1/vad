import numpy as np
from dataset.file_processing import load_csv


def load_files(files, features_num, dtype=np.float64):
    #
    #   Load features and labels from files
    #
    total_features = np.array([], dtype=dtype)
    total_labels = np.array([], dtype=dtype)

    for file, num in zip(files, features_num):
        features, labels = load_csv(file, num, dtype=dtype)

        total_features = np.concatenate([total_features, features])
        total_labels = np.concatenate([total_labels, labels])

        del features
        del labels

    return total_features, total_labels
