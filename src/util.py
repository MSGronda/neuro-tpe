import csv
import json
import os.path
import requests
import zipfile

DATASET_PATH = "../dataset"
PROCESSED_PATH = "../processed-data"
PROCESSED_FILE_FORMAT = "Processed.json"


def download_dataset():
    if dataset_already_exists():
        return

    print("\nDownloading dataset. This may take a bit...\n")

    url = "https://drive.usercontent.google.com/download?id=1T9xzczygDEIlnpubTg5UJGOoh5uj2QcQ&export=download&authuser=0&confirm=t&uuid=1bad2bc0-c7ef-45b1-8b96-467b13540475&at=APZUnTWR2ZH4AtPcT7baehHKJMPS:1716313011773"

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise ValueError("No pudo descargar el archivo")

    zip_file_path = f"{DATASET_PATH}/data.zip"

    with open(zip_file_path, 'wb') as file:
        for chunk in response.iter_content(1024):
            file.write(chunk)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_PATH)

    os.remove(zip_file_path)


def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)

        # Levantamos los canales del header
        for channel in next(reader):
            if channel != "":
                data.append((channel, []))

        # Levantamos la data para cada canal
        for row in reader:
            for i, value in enumerate(row):
                if value != "":
                    data[i][1].append(float(value))

    resp = {}
    for datum in data:
        resp[datum[0]] = datum[1]

    return resp


def load_dataset():
    data = []
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".csv"):
            data.append(load_data(f"{DATASET_PATH}/{filename}"))

    return data


def dump_processed_data(processed_dataset: []):
    j = 0
    for i, data_by_channel in enumerate(processed_dataset):
        with open(f"{PROCESSED_PATH}/S{j}G{i % 4}{PROCESSED_FILE_FORMAT}", "w") as file:
            json.dump(data_by_channel, file)

        if i % 4 == 0:
            j += 1


def load_processed_data():
    processed_data = []
    for filename in os.listdir(PROCESSED_PATH):
        if filename.endswith(PROCESSED_FILE_FORMAT):
            with open(f"{PROCESSED_PATH}/{filename}", "r") as file:
                processed_data.append(json.load(file))

    return processed_data


def dataset_already_exists():
    return os.path.exists(f"{DATASET_PATH}/S01G1AllChannels.csv")


def processed_data_already_exists():
    return os.path.exists(f"{PROCESSED_PATH}/S1G1{PROCESSED_FILE_FORMAT}")