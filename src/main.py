from sklearn.model_selection import train_test_split

from signal_processing import *
from dimension_reduction import *
from neural_network import *

if __name__ == '__main__':
    clip_length = 30        # En s. Particionamos el dataset en secciones para tener mas datos.
    sampling_rate = 128     # En Hz
    segment_length = 150    # Ventana que se usa en el welch
    n_components = 5        # Cantidad de components para PCA
    test_size = 0.2         # Relacion train-test para la red
    batch_size = 50         # Cantidad de datapoints para cada epoca
    epochs = 100            # Cantidad de epocas de training para la red

    # Obtenemos el dataset y lo procesamos (aplicando FFT)
    processed_dataset, dataset_classes = load_and_process_data(sampling_rate, segment_length, clip_length)

    # Opcional: hacer unos heatmaps que quedan lindos
    # heatmap(processed_dataset[0], FREQUENCY_BANDS_INDEX['Sigma'])

    # Aplicamos PCA
    matrices = convert_to_matrix(processed_dataset)
    transformed, _ = apply_pca(matrices, n_components)

    # Entrenamos la red
    X_train, X_test, y_train, y_test = train_test_split(transformed, generate_classes(transformed), test_size=test_size)
    model = train_model(X_train, y_train, batch_size, epochs)

    # Testeamos la red
    test_loss, test_acc = test_model(X_test, y_test, model)
    print('Test accuracy:', test_acc)

