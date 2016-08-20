import numpy as np
from dataset.file_processing import load_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import grid_search
from config import *


def load_files():
    #
    #   Load features and labels from files
    #
    voiced_features, voiced_res = load_csv(VOICED_FNAME, VOICED_FEATURES_NUM, dtype=np.float64)
    unvoiced_features, unvoiced_res = load_csv(UNVOICED_FNAME, UNVOICED_FEATURES_NUM, dtype=np.float64)

    features = np.concatenate([voiced_features, unvoiced_features])
    del voiced_features
    del unvoiced_features

    results = np.concatenate([voiced_res, unvoiced_res])
    del voiced_res
    del unvoiced_res

    return features, results


if __name__ == '__main__':
    cross_res_f = open('cross_validation.res', 'w')
    features, labels = load_files()
    labels = labels.reshape((len(labels),))

    depthes = np.arange(10, 41, 5)
    min_split = np.arange(7, 23, 5)
    min_leaves = np.arange(5, 21, 5)
    dt_parameters = {'max_depth': depthes, 'min_samples_leaf': min_leaves, 'min_samples_split': min_split}

    print("Begin crossvalidation on DT")
    dt = grid_search.GridSearchCV(DecisionTreeClassifier(), dt_parameters)
    dt.fit(features, labels)
    cross_res_f.write("Decision tree optimal : " + str(dt.best_params_) + " with score " + str(dt.best_score_) + "\r\n")

    cross_res_f.close()

"""
    kernel = ('poly', 'rbf', 'sigmoid')
    shrinking = (False, True)
    svm_parameters = {'kernel': kernel, 'shrinking': shrinking}
    print("Begin cross validation on SVC")
    svc = grid_search.GridSearchCV(SVC(), svm_parameters, n_jobs=4, cv=4)
    svc.fit(features, labels)
    cross_res_f.write("SVM optimal: " + str(svc.best_params_) + ' with score ' + str(svc.best_score_) + "\r\n")
"""
