import os
import numpy as np
import scipy.signal as sig
from src.util import dump_processed_data, PROCESSED_PATH, load_processed_data, download_dataset, load_dataset, \
    PROCESSED_FILE_FORMAT, processed_data_already_exists


EEG_LENGTH = 38253

# = = = = Procedimiento = = = =
# Siempre le pasamos los channels por separado
# Usar 6-8 canales (c3,c4, o3,o4, f3,f4, p3, p4, y los z)
# Cada tira de datos (de un canal), le aplico welch => te devuelve la transformada de fourier (frecuencia vs densidad espectral)
# Grafico lo que me devuelve
# Para cada rango, y divido en las distintas bandas (delta, theta, alpha, sigma...)
# Con las bandas, aplico area_psd a cada una => ese area es la potencia para ese ritmo, para ese canal.
# Tomo el area de toooodo y obtengo la potencia total.
# A cada potencia, lo divido por la potencia total => potencia relativa.


def load_and_process_data(sampling_rate: int, segment_length: int, clip_length: float):
    # Evitamos re-procesar los datos
    if processed_data_already_exists():
        return load_processed_data()

    download_dataset()

    partitioned_dataset, partitioned_classes = partition_dataset(*load_dataset(), seconds_to_size(clip_length))

    return process_data(partitioned_dataset, partitioned_classes, sampling_rate, segment_length)


def seconds_to_size(clip_length: float):
    return int((clip_length / (5 * 60)) * EEG_LENGTH)


def partition_dataset(dataset: [{}], classes: [], clip_size: int):
    partitioned_dataset = []
    partitioned_classes = []

    n_partitions = int(EEG_LENGTH / clip_size)

    for data_by_channel, data_class in zip(dataset, classes):
        partitions = [{} for _ in range(n_partitions)]
        partition_classes = [data_class for _ in range(n_partitions)]

        for p in range(n_partitions):
            for key, value in data_by_channel.items():      # O(n^3): un crimen contra la humanidad
                if p != n_partitions - 1:
                    partitions[p][key] = value[p * clip_size: (p+1) * clip_size]
                else:
                    partitions[p][key] = value[p * clip_size:]

        partitioned_dataset.extend(partitions)
        partitioned_classes.extend(partition_classes)

    return partitioned_dataset, partitioned_classes


def process_data(dataset: [], classes: [], sampling_rate: int, segment_length: int):
    print("Processing data...")

    # Evitamos re-procesar los datos
    if processed_data_already_exists():
        return load_processed_data()

    # Procesamos los datos
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

            data_by_channel[key] = relative_psd

        processed_dataset.append(data_by_channel)

    # Bajamos a disco
    dump_processed_data(processed_dataset, classes)

    return processed_dataset, classes


def apply_fft(signal: [], sampling_rate, segment_length):
    frequencies, psd_welch = sig.welch(signal, sampling_rate, nperseg=segment_length, nfft=segment_length)
    return frequencies, psd_welch


FREQUENCY_BANDS = [
    ('Delta', (0, 4)),
    ('Theta', (4, 7)),
    ('Alpha', (8, 12)),
    ('Beta', (12, 30)),
    ('Gamma', (30, 100)),
]
FREQUENCY_BANDS_INDEX = {
    'Delta': 0,
    'Theta': 1,
    'Alpha': 2,
    'Beta': 3,
    'Gamma': 4,
}
NAME = 0


RANGE = 1


def divide_bands(frequencies: [], psd: []):
    band_frequencies = [[] for _ in FREQUENCY_BANDS]
    band_psd = [[] for _ in FREQUENCY_BANDS]

    for i, band in enumerate(FREQUENCY_BANDS):
        for freq, ampl in zip(frequencies, psd):
            if band[RANGE][0] <= freq:
                if freq <= band[RANGE][1]:
                    band_frequencies[i].append(freq)
                    band_psd[i].append(ampl)
                else:
                    break    # Ya nos pasamos

    return band_frequencies, band_psd


def area_under_psd(frequencies: [],  psd: []):
    return np.trapz(psd, frequencies)
