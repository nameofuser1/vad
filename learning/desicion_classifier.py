import numpy as np

from learning.cross_validation import find_optimal_parameters

if __name__ == '__main__':
    depthes = np.arange(10, 41, 5)
    min_split = np.arange(7, 23, 5)
    min_leaves = np.arange(5, 21, 5)

    optimal = find_optimal_parameters(depthes, min_leaves, min_split)
    print(optimal)
