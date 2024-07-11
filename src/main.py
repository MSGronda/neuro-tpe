from sklearn.model_selection import train_test_split
from signal_processing import *
from dimension_reduction import *
from neural_network import *
from src.graphs import plot_first_layer_lrp, plot_relevance_network_graph, heatmap_lrp, plot_network_graph
from src.lrp import lrp, get_X_of_class_y, X_POS, normalize_r, normalized_matrix, average_lrp
from src.util import generate_component_labels, generate_channel_labels

if __name__ == '__main__':
    clip_length = 15        # En s. Particionamos el dataset en secciones para tener mas datos.
    sampling_rate = 128     # En Hz
    segment_length = 150    # Ventana que se usa en el welch
    n_components = 5        # Cantidad de components para PCA
    test_size = 0.2         # Relacion train-test para la red
    batch_size = 35         # Cantidad de datapoints para cada epoca
    epochs = 60             # Cantidad de epocas de training para la red
    lrp_class = 1           # Una de las 4 clases de datos (valores de y)

    # Obtenemos el dataset y lo procesamos (aplicando FFT)
    processed_dataset, dataset_classes = load_and_process_data(sampling_rate, segment_length, clip_length)

    # Opcional: hacer unos heatmaps que quedan lindos
    # heatmap(processed_dataset[0], FREQUENCY_BANDS_INDEX['Sigma'])

    # Aplicamos PCA
    matrices = convert_to_matrix(processed_dataset)
    transformed, _ = apply_pca(matrices, n_components)

    # Entrenamos la red
    X_train, X_test, y_train, y_test = train_test_split(transformed, dataset_classes, test_size=test_size)
    model = train_model_alt(X_train, y_train, batch_size, epochs)

    # Testeamos la red
    test_loss, test_acc = test_model(X_test, y_test, model)
    print('Test accuracy:', test_acc)

    # Graficamos la red
    # plot_network_graph(model)

    # Aplicamos el LRP
    test_subset = get_X_of_class_y(X_test, y_test, lrp_class)   # Elegimos datos de prueba de UNA clase
    lrp_results = [lrp(model, convert_to_single_array(test[X_POS])) for test in test_subset]

    for i in range(0, 4):
        # R = normalized_matrix(lrp_results[i][-1], n_components)

        # Graficamos la primera layer
        # for c in range(n_components):
        #     heatmap_lrp(R[:, c])
        # column_label, row_label = generate_component_labels(n_components), generate_channel_labels()
        # plot_first_layer_lrp(R, column_label, row_label)

        # Graficamos la red entera
        plot_relevance_network_graph(lrp_results[i], model)

    average, variance = average_lrp(lrp_results)

    plot_relevance_network_graph(average, model)
    plot_relevance_network_graph(variance, model, 'blue')
