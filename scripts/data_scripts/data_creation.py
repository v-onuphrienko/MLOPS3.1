import os

import numpy as np
import pandas as pd


# Функция генерации данных
def gen_data(size):
    data = pd.DataFrame({
        "temp": np.random.normal(loc=20, scale=5, size=size),
        "air_humidity": np.random.normal(loc=50, scale=10, size=size),
        "pressure": np.random.normal(loc=1000, scale=50, size=size)
    })

    return data

# Функция добавления шума
def add_noise(data, loc=0, scale=2, size=2500):
    data = data + np.random.normal(loc=loc, scale=scale, size=size)

# Функция добавления метки
def add_mark(data):
    data["label"] = np.where(data["temp"] > 25, 1, 0)

# Функция сохранения данных
def save_data(data, dir_name, file_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(f"{dir_name}/{file_name}.csv", index=False)

def init_data(name, size, with_noize=False):
    data = gen_data(size)

    if with_noize:
        add_noise(data["temp"], size=size)
        add_noise(data["air_humidity"], size=size)
        add_noise(data["pressure"], size=size)

    add_mark(data)

    save_data(data, "~/project/MLOPS3.1/data/raw", name)


# Учебные данные
init_data("train", 2500, True)

# Тестовые данные
init_data("test", 1000)