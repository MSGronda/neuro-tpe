import numpy as np
from keras import Sequential, layers


def convert_to_array(dataset: [np.array]):
    return np.array([data.flatten() for data in dataset]).astype("float32")


def generate_classes(dataset: []):
    return [i % 4 for i in range(len(dataset))]


def train_model(X_train, y_train, batch_size: int, epochs: int):

    X = convert_to_array(X_train)
    y = np.array(y_train).astype("float32")

    model = Sequential([
        layers.Input(shape=X[0].shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')
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


