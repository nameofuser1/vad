import numpy as np
from sklearn import grid_search
from sklearn.tree import DecisionTreeClassifier
from utils import load_files

import os
import sys

sys.path.insert(0, os.path.abspath('../'))
from config import *


if __name__ == '__main__':
    cross_res_f = open('./cross_validation1.res', 'w')
    features, labels = load_files([VOICED_FNAME, UNVOICED_FNAME], [VOICED_FEATURES_NUM, UNVOICED_FEATURES_NUM])
    labels = labels.reshape((len(labels),))
    print(len(features))
    print(len(labels))

    depthes = np.arange(10, 41, 5)
    min_split = np.arange(7, 23, 5)
    min_leaves = np.arange(5, 21, 5)
    dt_parameters = {'max_depth': depthes, 'min_samples_leaf': min_leaves, 'min_samples_split': min_split}

    print("Begin cross validation on DT")
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
