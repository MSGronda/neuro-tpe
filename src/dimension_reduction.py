import numpy as np


def convert_to_matrix(dataset: [{}]):
    resp = []
    for data_by_channel in dataset:
        matrix = []
        for row in data_by_channel.values():
            matrix.append(row)
        resp.append(np.array(matrix))

    return resp
