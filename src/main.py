from sklearn.model_selection import train_test_split

from util import *
from signal_processing import *
from graphs import *
from dimension_reduction import *
from neural_network import *

if __name__ == '__main__':
    sampling_rate = 128     # En Hz
    segment_length = 150    # Ventana que se usa en el welch
    n_components = 3        # Cantidad de components para PCA

    # Obtenemos el dataset y lo procesamos (aplicando FFT)
    processed_dataset = load_and_process_data(sampling_rate, segment_length)

    # Opcional: hacer unos heatmaps que quedan lindos
    # heatmap(processed_dataset[0], FREQUENCY_BANDS_INDEX['Sigma'])

    # Aplicamos PCA
    matrices = convert_to_matrix(processed_dataset)
    transformed, _ = apply_pca(matrices, n_components)

    # Entrenamos la red
    X_train, X_test, y_train, y_test = train_test_split(transformed, generate_classes(transformed), test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, 32, 10)

    test_loss, test_acc = test_model(X_test, y_test, model)
    print('Test accuracy:', test_acc)

