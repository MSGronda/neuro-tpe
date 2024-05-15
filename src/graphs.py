import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import plot_topomap


def plot_signal(x: [], y: [], x_label: str, y_label: str):
    plt.plot(x, y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


def heatmap(data_by_channel: {}, frequency_band: int):
    channels = [channel for channel in data_by_channel.keys()]
    psd_for_frequency = [psd[frequency_band] for psd in data_by_channel.values()]

    montage = mne.channels.make_standard_montage('standard_1020')
    pos = np.array([montage.dig[i]['r'][:2] for i, ch in enumerate(montage.ch_names) if ch in channels])

    mne.viz.plot_topomap(
        psd_for_frequency,
        pos,
        names=montage.ch_names,
        cmap='viridis',
        res=500,
        size=8,
        contours=8,
        sensors=True,
        # sphere=(0, 0, 0, 0.1)
    )
