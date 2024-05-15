from util import *
from signal_processing import *
from graphs import *


if __name__ == '__main__':
    sampling_rate = 128
    segment_length = 150

    # Obtenemos el dataset y lo cargamos en memoria
    download_dataset()
    dataset = load_all_data()

    # Para cada canal aplicamos FFT y obtenemos las potencias relativas
    for key, value in dataset[0].items():
        freq, psd = apply_fft(value, sampling_rate, segment_length)
        plot_signal(freq, psd, "Frecuencia (Hz)", f"Amplitud (PSD) - Canal {key}")

        band_freq, band_psd = divide_bands(freq, psd)
        total_area_psd = area_under_psd(freq, psd)

        relative_powers = []

        for freq, psd in zip(band_freq, band_psd):
            relative_powers.append(area_under_psd(freq, psd) / total_area_psd)

        print(relative_powers)
