import csv
import os.path
import requests

DATASET_PATH = "../dataset"


def download_dataset():
    if os.path.exists(f"{DATASET_PATH}/g1.csv"):
        return

    urls = [
        "https://drive.usercontent.google.com/download?id=17aJhaN1x2roMco2o6qeOc_ib5v9QjEPx&export=download&confirm=t&uuid=036070f5-8609-4301-91a3-50fbd0ef656d",
        "https://drive.usercontent.google.com/download?id=1h7u1CXjDOjpcnJyccHCdGjPg4bLHhx-E&export=download&authuser=0&confirm=t&uuid=1ee3eeef-5079-4314-8ed4-1d38ee8c0e66&at=APZUnTX_UEFJxMDZAhwKJlawe5TG:1715699364669",
        "https://drive.usercontent.google.com/download?id=1c9ur3yTcvNeM3M4mXlZgWRTuTTctyc-V&export=download&authuser=0&confirm=t&uuid=c3525b2f-7e4b-41f5-9927-014d99f4cf24&at=APZUnTX1ZWya-4vEyqZOWI0_YHzs:1715699419409",
        "https://drive.usercontent.google.com/download?id=1i8p4ADsczp7iCZNOR-L5FPrNy9byeIdl&export=download&authuser=0&confirm=t&uuid=12149e6e-3fb6-4b4d-8e56-7725235848ec&at=APZUnTWVWLWsEov80jTaXd6U6Uc4:1715699446553"
    ]

    for i, url in enumerate(urls):
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            raise ValueError("No pudo descargar el archivo")

        with open(f"{DATASET_PATH}/g{i+1}.csv", 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)


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


def load_all_data():
    data = []
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".csv"):
            data.append(load_data(f"{DATASET_PATH}/{filename}"))

    return data


