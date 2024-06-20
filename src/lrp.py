from keras import Model
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def get_outputs(model, X):
    outputs = [X]
    intermediate_layers = [Model(inputs=model.input, outputs=layer.output) for layer in model.layers]

    for i in range(0, len(intermediate_layers) - 1):
        outputs.append(intermediate_layers[i].predict(X))

    return outputs


def lrp_dense(layer, R, inputs):
    W = layer.get_weights()[0]
    V = np.maximum(W, 0.0)
    Z = np.dot(inputs, V) + 1e-9 * np.sign(np.dot(inputs, V))
    S = R / Z
    C = np.dot(S, np.transpose(V))
    return inputs * C


def lrp(model, x):

    outputs = get_outputs(model, x)

    R = model.predict(x)
    layer_R = [R]

    # Iterate through the layers in reverse order and apply LRP
    for i in range(len(model.layers) - 1, 0, -1):
        R = lrp_dense(model.layers[i], R, outputs[i])
        layer_R.append(R)

    return layer_R

# = = = = = = = = =  GRAPHING = = = = = = = = =


def normalize_r(layer_R):
    relevance_scores = []
    for layer in reversed(layer_R):
        layer = layer.flatten()
        min_weight = min(layer)
        max_weight = max(layer)
        relevance_scores.append([(r - min_weight) / (max_weight - min_weight) for r in layer])

    return relevance_scores


def flatten_irregular(relevance_scores):
    resp = []
    for layer in relevance_scores:
        resp.extend(layer)
    return resp


def plot_network_graph(layer_R, model):
    figsize = (40, 40)

    G = nx.DiGraph()
    layer_sizes = [layer.units for layer in model.layers if 'dense' in layer.name]
    layer_sizes.insert(0, model.input.shape[1])

    relevance_scores = normalize_r(layer_R)

    max_neurons = max(layer_sizes)

    for i, size in enumerate(layer_sizes):
        for j in range(size):
            G.add_node(f'{i}_{j}', layer=i, pos=(i, (max_neurons / (layer_sizes[i] + 2)) * (j+1)))  # Super secret sauce that give cool spacing

    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                G.add_edge(f'{i}_{j}', f'{i+1}_{k}', weight=(relevance_scores[i][j] + relevance_scores[i+1][k]) / 2)

    edges = G.edges(data=True)
    nodes = G.nodes(data=True)

    pos = {node: (data['layer'], data['pos'][1]) for node, data in nodes}

    plt.figure(figsize=figsize)

    for (u, v), alpha in zip(nodes, flatten_irregular(relevance_scores)):
        nx.draw_networkx_nodes(G, pos, nodelist=[u], node_size=300, alpha=alpha, node_color='red', linewidths=1, edgecolors='black')

    for (u, v, d), alpha in zip(edges, [edge[2]['weight'] for edge in edges]):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1, alpha=alpha, edge_color='red', arrows=False)

    plt.title('Neural Network Graph with Relevance Scores')
    plt.tight_layout()
    plt.show()