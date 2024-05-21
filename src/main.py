from util import *
from signal_processing import *
from graphs import *
from dimension_reduction import *


if __name__ == '__main__':
    sampling_rate = 128     # En Hz
    segment_length = 150    # Ventana que se usa en el welch
    n_components = 3        # Cantidad de components para PCA

    # Obtenemos el dataset y lo procesamos (aplicando FFT)
    processed_dataset = load_and_process_data(sampling_rate, segment_length)

    # Opcional: hacer unos heatmaps que quedan lindos
    # heatmap(processed_dataset[0], FREQUENCY_BANDS_INDEX['Sigma'])

    matrices = convert_to_matrix(processed_dataset)

    transformed, _ = apply_pca(matrices, n_components)

    print(transformed)

