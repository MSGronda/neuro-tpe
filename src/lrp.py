import random

from keras import Model
import numpy as np


X_POS = 0
Y_POS = 1


def get_outputs(model, X):
    outputs = [X]
    intermediate_layers = [Model(inputs=model.input, outputs=layer.output) for layer in model.layers]

    for i in range(0, len(intermediate_layers) - 1):
        outputs.append(intermediate_layers[i].predict(X))

    return outputs


def lrp_dense(layer, R, inputs):
    W = layer.get_weights()[0]
    V = np.maximum(W, 0.0)
    Z = np.dot(inputs, V) + 1e-9 * np.sign(np.dot(inputs, V))
    S = R / Z
    C = np.dot(S, np.transpose(V))
    return inputs * C


def lrp(model, x):

    outputs = get_outputs(model, x)

    R = model.predict(x)
    layer_R = [R]

    # Iterate through the layers in reverse order and apply LRP
    for i in range(len(model.layers) - 1, 0, -1):
        R = lrp_dense(model.layers[i], R, outputs[i])
        layer_R.append(R)

    return layer_R


def get_X_of_class_y(X, Y, c, limit=20):
    resp = []

    for x,y in zip(X, Y):
        if y == c:
            resp.append((x, y))

    random.shuffle(resp)

    return resp[:limit]


def normalize_r(layer_R):
    relevance_scores = []
    for layer in reversed(layer_R):
        layer = layer.flatten()
        min_weight = min(layer)
        max_weight = max(layer)
        relevance_scores.append([(r - min_weight) / (max_weight - min_weight) for r in layer])

    return relevance_scores


def flatten_irregular(relevance_scores):
    resp = []
    for layer in relevance_scores:
        resp.extend(layer)
    return resp


def normalized_matrix(R, n_components):     # un asquete
    return np.reshape(np.array(normalize_r([R])), (int(R.shape[1] / n_components), n_components))
