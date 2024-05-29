import numpy as np
from keras import Sequential, layers
from keras.src.layers import SimpleRNN, Dense
from sklearn.preprocessing import OneHotEncoder

NUM_CLASSES = 4


def convert_to_array(dataset: [np.array]):
    return np.array([data.flatten() for data in dataset]).astype("float32")


def generate_classes(dataset: []):
    return [i % NUM_CLASSES for i in range(len(dataset))]


def train_model(X_train, y_train, batch_size: int, epochs: int):

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




