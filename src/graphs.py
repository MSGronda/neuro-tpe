import matplotlib.pyplot as plt


def plot_signal(x: [], y: [], x_label: str, y_label: str):
    plt.plot(x, y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()
