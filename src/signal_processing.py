import numpy as np
import scipy.signal as sig

# = = = = Procedimiento = = = =
# Siempre le pasamos los channels por separado
# Usar 6-8 canales (c3,c4, o3,o4, f3,f4, p3, p4, y los z)
# Cada tira de datos (de un canal), le aplico welch => te devuelve la transformada de fourier (frecuencia vs densidad espectral)
# Grafico lo que me devuelve
# Para cada rango, y divido en las distintas bandas (delta, theta, alpha, sigma...)
# Con las bandas, aplico area_psd a cada una => ese area es la potencia para ese ritmo, para ese canal.
# Tomo el area de toooodo y obtengo la potencia total.
# A cada potencia, lo divido por la potencia total => potencia relativa.


def apply_fft(signal: [], sampling_rate, segment_length):
    frequencies, psd_welch = sig.welch(signal, sampling_rate, nperseg=segment_length, nfft=segment_length)
    return frequencies, psd_welch


FREQUENCY_BANDS = [
    ('Delta', (0.5, 4)),
    ('Theta', (4, 8)),
    ('Alpha', (8, 13)),
    ('Sigma', (13, 15)),
    ('Beta', (15, 30)),
    ('Gamma', (30, 100)),
]
FREQUENCY_BANDS_INDEX = {
    'Delta': 0,
    'Theta': 1,
    'Alpha': 2,
    'Sigma': 3,
    'Beta': 4,
    'Gamma': 5,
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

