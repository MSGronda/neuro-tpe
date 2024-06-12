from keras import backend as K, Model
import tensorflow as tf
import numpy as np


def get_outputs(model, X):
    model.predict(X)

    outputs = []
    intermediate_layers = [Model(inputs=model.input, outputs=layer) for layer in model.layers]
    prev_x = X

    for layer in intermediate_layers:
        prev_x = layer.predict(X)

        outputs.append(prev_x)

    return outputs


def lrp_dense(layer, R):
    W = layer.get_weights()[0]
    V = tf.maximum(W, 0.0)
    Z = tf.matmul(layer.input, V) + 1e-9
    S = R / Z
    C = tf.matmul(S, tf.transpose(V))
    return layer.input * C


def lrp(model, x):

    outputs = get_outputs(model, x)

    R = x

    # Iterate through the layers in reverse order and apply LRP
    for layer in reversed(model.layers):
        R = lrp_dense(layer, R)

    return R

