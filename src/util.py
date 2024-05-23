import csv
import json
import os.path
import requests
import zipfile

DATASET_PATH = "../dataset"
PROCESSED_PATH = "../processed-data"
PROCESSED_FILE_FORMAT = "processed.json"


def download_dataset():
    if dataset_already_exists():
        return

    print("Downloading dataset. This may take a bit...")

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
    print("Loading dataset...")
    data = []
    classes = []
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".csv"):
            data.append(load_data(f"{DATASET_PATH}/{filename}"))
            classes.append(int(filename[4]) - 1)

    return data, classes


def dump_processed_data(processed_dataset: [], classes: []):
    with open(f"{PROCESSED_PATH}/{PROCESSED_FILE_FORMAT}", "w") as file:
        json.dump({"processed": processed_dataset, "classes": classes}, file)


def load_processed_data():
    with open(f"{PROCESSED_PATH}/{PROCESSED_FILE_FORMAT}", "r") as file:
        obj = json.load(file)

        return obj["processed"], obj["classes"]


def dataset_already_exists():
    return os.path.exists(f"{DATASET_PATH}/S01G1AllChannels.csv")


def processed_data_already_exists():
    return os.path.exists(f"{PROCESSED_PATH}/{PROCESSED_FILE_FORMAT}")