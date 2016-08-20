from sklearn.cross_validation import cross_val_score
from utils import cartesian


def find_optimal_parameters(cls, features, labels,  **kwargs):
    keys = []
    classifier_parameters_list = []

    for key in kwargs:
        keys.append(key)
        classifier_parameters_list.append(kwargs.get(key))

    combinations = cartesian(classifier_parameters_list)
    scores = []

    print('Begin cross-validation on ' + cls.__name__)
    for i in range(len(combinations)):
        classifier_params = {}

        for j in range(len(keys)):
            classifier_params[keys[j]] = combinations[i][j]

        print(labels.dtype)
        mean_score = cross_val_score(cls(**classifier_params),
                                     features, labels, cv=4, n_jobs=-1, verbose=2).mean()

        classifier_params['score'] = mean_score
        print('Classifier params: ' + str(classifier_params))
        scores.append(classifier_params)

    optimal_parameters = {'score': 0}
    for res in scores:
        if res['score'] > optimal_parameters['score']:
            optimal_parameters = res

    return optimal_parameters
