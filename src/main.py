from util import *
from signal_processing import *
from graphs import *

if __name__ == '__main__':
    sampling_rate = 128     # En Hz
    segment_length = 150    # Ventana que se usa en el welch

    # Obtenemos el dataset y lo cargamos en memoria
    download_dataset()
    dataset = load_dataset()

    # Para cada canal aplicamos FFT y obtenemos las potencias relativas
    processed_dataset = process_data(dataset, sampling_rate, segment_length)

    heatmap(processed_dataset[0], FREQUENCY_BANDS_INDEX['Sigma'])




