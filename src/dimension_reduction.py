import numpy as np
from sklearn.decomposition import PCA


def convert_to_matrix(dataset: [{}]):
    resp = []
    for data_by_channel in dataset:
        matrix = []
        for row in data_by_channel.values():
            matrix.append(row)
        resp.append(np.array(matrix))

    return resp


def apply_pca(dataset: [np.array], n_components=None):
    transformed = []
    explained_variance_ratios = []

    for matrix in dataset:

        # TODO CHECK: sobre que hacemos la reduccion, los canales o las potencias relativas ?

        if n_components is None:
            pca = PCA(n_components=len(matrix[0]))
        else:
            pca = PCA(n_components)

        pca.fit(matrix)

        transformed.append(pca.transform(matrix))
        explained_variance_ratios.append(pca.explained_variance_ratio_)

    return transformed, explained_variance_ratios
