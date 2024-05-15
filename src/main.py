from util import *
from signal_processing import *
from graphs import *

if __name__ == '__main__':
    sampling_rate = 128     # En Hz
    segment_length = 150    # Ventana que se usa en el welch

    # Obtenemos el dataset y lo cargamos en memoria
    download_dataset()
    dataset = load_all_data()

    # Para cada canal aplicamos FFT y obtenemos las potencias relativas
    processed_dataset = []
    for data in dataset:
        data_by_channel = {}
        for key, value in data.items():
            freq, psd = apply_fft(value, sampling_rate, segment_length)
            band_freq, band_psd = divide_bands(freq, psd)
            total_area_psd = area_under_psd(freq, psd)

            relative_psd = []

            for freq, psd in zip(band_freq, band_psd):
                relative_psd.append(area_under_psd(freq, psd) / total_area_psd)

            data_by_channel[key] = np.array(relative_psd)

        processed_dataset.append(data_by_channel)

    heatmap(processed_dataset[0], FREQUENCY_BANDS_INDEX['Sigma'])




