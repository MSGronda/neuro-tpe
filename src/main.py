from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from signal_processing import *
from dimension_reduction import *
from neural_network import *

if __name__ == '__main__':
    clip_length = 15        # En s. Particionamos el dataset en secciones para tener mas datos.
    sampling_rate = 128     # En Hz
    segment_length = 150    # Ventana que se usa en el welch
    n_components = 3        # Cantidad de components para PCA
    test_size = 0.2         # Relacion train-test para la red
    batch_size = 35         # Cantidad de datapoints para cada epoca
    epochs = 60             # Cantidad de epocas de training para la red

    # Obtenemos el dataset y lo procesamos (aplicando FFT)
    processed_dataset, dataset_classes = load_and_process_data(sampling_rate, segment_length, clip_length)

    # Opcional: hacer unos heatmaps que quedan lindos
    # heatmap(processed_dataset[0], FREQUENCY_BANDS_INDEX['Sigma'])

    # Aplicamos PCA
    matrices = convert_to_matrix(processed_dataset)
    transformed, _ = apply_pca(matrices, n_components)

    # Entrenamos la red
    X_train, X_test, y_train, y_test = train_test_split(transformed, dataset_classes, test_size=test_size)
    model = train_model(X_train, y_train, batch_size, epochs)

    # Entrenamos la red recurrente (que no funciona muy bien)
    # X, y = sequentialize_data(transformed, dataset_classes)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # model = train_model_rnn(X_train, y_train, epochs)

    # Testeamos la red
    test_loss, test_acc = test_model(X_test, y_test, model)
    print('Test accuracy:', test_acc)

