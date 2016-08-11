from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from dataset.file_processing import load_files


def find_optimal_parameters(depthes, samples_leaf, samples_split):
    features, labels = load_files()
    print(features.shape)
    labels = labels.reshape((len(labels),))
    print(labels.shape)
    combinations = cartesian((depthes, samples_leaf, samples_split))
    scores = []

    print('Begin cross-validation')
    for i in range(len(combinations)):
        depth = combinations[i][0]
        min_samples_leaf = combinations[i][1]
        min_samples_split = combinations[i][2]

        mean_score = cross_val_score(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples_leaf,
                                                            min_samples_split=min_samples_split),
                                     features, labels, cv=4, n_jobs=4, verbose=2).mean()

        res = {'score': mean_score, 'leaf': min_samples_leaf, 'split': min_samples_split, 'depth':depth}
        scores.append(res)

    optimal_parameters = {'score': 9999999}
    for res in scores:
        if res['score'] < optimal_parameters['score']:
            optimal_parameters = res

    return optimal_parameters