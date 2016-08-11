from sklearn.cluster import SpectralClustering
from learning import prepare_dataset


if __name__ == '__main__':

    train_features, test_features, train_res, test_res = prepare_dataset()

    clustering = SpectralClustering(n_clusters=3)
    print("Begin clustering")
    res = clustering.fit_predict(train_features, test_features)
    print(res.shape)






