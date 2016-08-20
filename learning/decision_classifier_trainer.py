from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from utils import load_files
from config import *
import cPickle

if __name__ == "__main__":
    print('Loading files...')
    features, labels = load_files()
    labels = labels.reshape((len(labels),))
    print('Splitting into test and train datasets')
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                train_size=TRAIN_TEST_RATIO)
    print('Train features: ' + str(len(train_features)))
    print('Train labels: ' + str(len(train_labels)))
    print('Test features: ' + str(len(test_features)))
    print('Test labels: ' + str(len(test_labels)))
    print('')

    params = {'min_samples_split': 22, 'max_depth': 25, 'min_samples_leaf': 20}
    cls = DecisionTreeClassifier(**params)

    print('Training...')
    cls.fit(train_features, train_labels)

    print('Saving classifier...')
    cls_f = open('../classifiers/decision_classifier.cls', 'w')
    cPickle.dump(cls, cls_f)
    cls_f.close()

    print('Calculating score...')
    score = cls.score(test_features, test_labels)

    print('Saving results...')
    f = open('decision.res', 'w')
    f.write('Decision tree with parameters: ' + str(params))
    f.write('Score: ' + str(score))
    f.write('Used voiced features: ' + str(VOICED_FEATURES_NUM))
    f.write('Used unvoiced features: ' + str(UNVOICED_FEATURES_NUM))
    f.close()

    print('Fuck yeaahh...')
