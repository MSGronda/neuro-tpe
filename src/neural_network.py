import logging

import numpy as np
from keras import Sequential, layers, Model
import tensorflow as tf

NUM_CLASSES = 4

tf.get_logger().setLevel(logging.ERROR)     # Asi no rompe los huevos con los warnings


def convert_to_array(dataset: [np.array]):
    return np.array([data.flatten() for data in dataset]).astype("float32")


def convert_to_single_array(x: np.array):
    return np.array(x.flatten()).reshape(1, -1)


def generate_classes(dataset: []):
    return [i % NUM_CLASSES for i in range(len(dataset))]


def train_model(X_train, y_train, batch_size: int, epochs: int):

    # NO USAR ESTA VERSION. Por mas que parezca que es lo mismo que train_model_alt
    # Si usas esta, el LRP explota.

    X = convert_to_array(X_train)
    y = np.array(y_train).astype("float32")

    model = Sequential([
        layers.Input(shape=X[0].shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
    )

    model.fit(X, y, batch_size=batch_size, epochs=epochs)

    return model


def train_model_alt(X_train, y_train, batch_size: int, epochs: int):

    X = convert_to_array(X_train)
    y = np.array(y_train).astype("float32")

    input_layer = layers.Input(shape=X[0].shape)
    i1 = layers.Dense(64, activation='relu')(input_layer)
    i2 = layers.Dense(32, activation='relu')(i1)
    i3 = layers.Dense(16, activation='relu')(i2)
    i4 = layers.Dense(8, activation='relu')(i3)
    output_layer = layers.Dense(NUM_CLASSES, activation='softmax')(i4)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
    )

    model.fit(X, y, batch_size=batch_size, epochs=epochs)

    return model


def test_model(X_test, y_test, model):
    X = convert_to_array(X_test)
    y = np.array(y_test).astype("float32")

    return model.evaluate(X, y)


def get_sequence_length(y_train):
    prev = y_train[0]
    for i in range(len(y_train)):
        if prev != y_train[i]:
            return i
    return -1


def sequentialize_data(X_data, y_data):

    seq_length = get_sequence_length(y_data)
    X = []

    for i in range(int(len(X_data) / seq_length)):
        if i == int(len(X_data) / seq_length):
            X.append([x.flatten() for x in X_data[i * seq_length:]])    # O(n!), buenisimo
        else:
            X.append([x.flatten() for x in X_data[i * seq_length:(i+1) * seq_length]])    # O(n!), buenisimo

    return np.array(X), np.array(generate_classes(X))


def train_model_rnn(X_train, y_train, epochs):

    model = Sequential([
        layers.Input(shape=X_train[0].shape),
        layers.SimpleRNN(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
  )

    model.fit(X_train, y_train, epochs=epochs)

    return model


def normalize_weights(model):
    # Estoy convencido que este es el peor codigo que escribi en mi vida

    weights = [weight for weight in model.get_weights() if len(weight.shape) != 1]

    min_weight = np.min([item for sublist1 in weights for sublist2 in sublist1 for item in sublist2])
    max_weight = np.max([item for sublist1 in weights for sublist2 in sublist1 for item in sublist2])

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                w = weights[i][j][k]
                if w > 0:
                    weights[i][j][k] = weights[i][j][k] / max_weight
                else:
                    weights[i][j][k] = (weights[i][j][k] - min_weight) / min_weight

    return weights




