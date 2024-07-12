# Layer-Wise Relevance Propagation - Neurociencia Computacional
En este trabajo práctico final, se implementó el algoritmo de Layer-Wise Relevance Propagation con el fin de observar la manera que un multi-layer perceptron "toma" decisiones. 
También se aplicaron los siguientes métodos: Fast-Fourier Transform (FFT), Principal Component Analysis (PCA).

## Requisitos
1. Python 3
2. Pip

## Ejecución de proyecto
1. Primero deberán instalar las librarians de python de la siguiente forma:
```shell
pip install requirements.txt
```
2. Ejecutar el archivo main:
```shell
python ./src/main.py
```

## Dataset
El dataset consiste de mediciones EEG de 14 canales, durante las cuales a los participantes se les mostraba gameplay de 4 videojuegos distintos. 
Este se debería descargar automáticamente. En el caso de que no se haga, se puede obtener en el siguiente link:
https://www.kaggle.com/datasets/sigfest/database-for-emotion-recognition-system-gameemo

## Funcionamiento
En el archivo `main.py`, se definen las siguientes constantes, las cuales se pueden modificar para alterar el funcionamiento del mismo:
```python
clip_length = 15        # En s. Particionamos el dataset en secciones para tener mas datos.
sampling_rate = 128     # En Hz
segment_length = 150    # Ventana que se usa en el welch
n_components = 5        # Cantidad de components para PCA
test_size = 0.2         # Relacion train-test para la red
batch_size = 35         # Cantidad de datapoints para cada epoca
epochs = 60             # Cantidad de epocas de training para la red
lrp_class = 1           # Una de las 4 clases de datos (0, 1, 2, 3)
```
A continuación se enuncia los pasos que sigue el programa:
1. Descarga el dataset y lo guarda en el directorio `dataset`
2. Lo procesa, aplicando FFT guardando los resultados en `processed-data/processed.json`. Si se ejecuta el programa nuevamente, se usara los resultados del archivo, evitando reprocesar el dataset.
3. Aplica PCA
4. Genera la red neuronal y la entrena
5. Aplica LRP y genera los gráficos relevantes.

