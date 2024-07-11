import math

import mne
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from src.constants import CHANNELS
from src.lrp import normalize_r, flatten_irregular
from src.neural_network import normalize_weights


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


def average_edge_relevance(relevance_scores, i, j, k):
    return (relevance_scores[i][j] + relevance_scores[i + 1][k]) / 2


def minimized_edge_relevance(relevance_scores, i, j, k, scale):
    relevance = average_edge_relevance(relevance_scores, i, j, k)
    return min(max(math.exp(scale * relevance - scale) - math.pow(math.e, -scale), 0), 1)


def decaying_edge_relevance(relevance_scores, i, j, k):
    relevance = average_edge_relevance(relevance_scores, i, j, k)
    return relevance * ((i+1) / (len(relevance_scores)))


def plot_network_graph(model):
    figsize = (40, 40)

    G = nx.DiGraph()
    layer_sizes = [layer.units for layer in model.layers if 'dense' in layer.name]
    layer_sizes.insert(0, model.input.shape[1])

    weights = normalize_weights(model)

    max_neurons = max(layer_sizes)

    for i, size in enumerate(layer_sizes):
        for j in range(size):
            G.add_node(f'{i}_{j}', layer=i, pos=(i, (max_neurons / (layer_sizes[i] + 2)) * (j + 1)))  # Super secret sauce that give cool spacing

    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                G.add_edge(f'{i}_{j}',f'{i + 1}_{k}',weight=weights[i][j][k])

    edges = G.edges(data=True)
    nodes = G.nodes(data=True)

    pos = {node: (data['layer'], data['pos'][1]) for node, data in nodes}

    plt.figure(figsize=figsize)

    for u, v in nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=[u], node_size=300, node_color='black', linewidths=1)

    for (u, v, d), alpha in zip(edges, [edge[2]['weight'] for edge in edges]):
        if alpha > 0:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1, alpha=alpha, edge_color='green', arrows=False)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1, alpha=abs(alpha), edge_color='orange', arrows=False)

    plt.tight_layout()
    plt.show()


def plot_relevance_network_graph(layer_R, model):
    figsize = (40, 40)

    G = nx.DiGraph()
    layer_sizes = [layer.units for layer in model.layers if 'dense' in layer.name]
    layer_sizes.insert(0, model.input.shape[1])

    relevance_scores = normalize_r(layer_R)

    max_neurons = max(layer_sizes)

    for i, size in enumerate(layer_sizes):
        for j in range(size):
            G.add_node(f'{i}_{j}', layer=i, pos=(
            i, (max_neurons / (layer_sizes[i] + 2)) * (j + 1)))  # Super secret sauce that give cool spacing

    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                G.add_edge(
            f'{i}_{j}',
            f'{i + 1}_{k}',
                    weight=(minimized_edge_relevance(relevance_scores, i, j, k, 6))
                )

    edges = G.edges(data=True)
    nodes = G.nodes(data=True)

    pos = {node: (data['layer'], data['pos'][1]) for node, data in nodes}

    plt.figure(figsize=figsize)

    for (u, v), alpha in zip(nodes, flatten_irregular(relevance_scores)):
        nx.draw_networkx_nodes(G, pos, nodelist=[u], node_size=300, alpha=alpha, node_color='red', linewidths=1,
                               edgecolors='black')

    for (u, v, d), alpha in zip(edges, [edge[2]['weight'] for edge in edges]):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1, alpha=alpha, edge_color='red', arrows=False)

    plt.tight_layout()
    plt.show()


def plot_first_layer_lrp(R, column_labels, row_labels):
    plt.figure(figsize=(5, 5))
    plt.imshow(R, cmap='viridis', interpolation='none',  aspect='auto')
    plt.colorbar()

    plt.xticks(ticks=np.arange(len(column_labels)), labels=column_labels)
    plt.yticks(ticks=np.arange(len(row_labels)), labels=row_labels)

    plt.tight_layout()
    plt.show()


def heatmap_lrp(R):
    channels = CHANNELS
    psd_for_frequency = R

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