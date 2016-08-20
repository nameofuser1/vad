import numpy as np

from config import VOICED_FNAME, VOICED_FEATURES_NUM, UNVOICED_FNAME, UNVOICED_FEATURES_NUM
from dataset.file_processing import load_csv


def load_files():
    #
    #   Load features and labels from files
    #
    voiced_features, voiced_res = load_csv(VOICED_FNAME, VOICED_FEATURES_NUM, dtype=np.float64)
    unvoiced_features, unvoiced_res = load_csv(UNVOICED_FNAME, UNVOICED_FEATURES_NUM, dtype=np.float64)

    features = np.concatenate([voiced_features, unvoiced_features])
    del voiced_features
    del unvoiced_features

    labels = np.concatenate([voiced_res, unvoiced_res])
    del voiced_res
    del unvoiced_res

    return features, labels