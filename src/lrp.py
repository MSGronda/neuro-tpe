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
    Z = np.dot(inputs, V) + 1e-9
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


def plot_network_graph(layer_R, model):
    figsize = (40, 40)

    G = nx.DiGraph()
    layer_sizes = [layer.units for layer in model.layers if 'dense' in layer.name]
    layer_sizes.insert(0, model.input.shape[1])

    relevance_scores = [r.flatten() for r in reversed(layer_R)]

    max_neurons = max(layer_sizes)

    for i, size in enumerate(layer_sizes):
        for j in range(size):
            G.add_node(f'{i}_{j}', layer=i, pos=(i, (max_neurons / (layer_sizes[i] + 2)) * (j+1)))  # Super secret sauce that give cool spacing
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                G.add_edge(f'{i}_{j}', f'{i+1}_{k}', weight=relevance_scores[i][j])

    pos = {node: (data['layer'], data['pos'][1]) for node, data in G.nodes(data=True)}

    # Extract edge weights for opacity
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    # Normalize weights for opacity (0 to 1)
    min_weight = min(weights)
    max_weight = max(weights)
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]

    # Draw the network graph
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')

    # Draw edges with varying opacity
    for (u, v, d), alpha in zip(edges, normalized_weights):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red', alpha=alpha, width=1)

    plt.title('Neural Network Graph with Relevance Scores')
    plt.tight_layout()
    plt.show()