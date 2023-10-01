import sys
import os
import yaml

import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression


if len(sys.argv) != 3:
    sys.stderr.write("Ошибка в аргументах. Пример выполнения:\n")
    sys.stderr.write("python model_preparation.py data_file model\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("models", sys.argv[2])

params = yaml.safe_load(open("params.yaml"))["train"]
p_seed = params["seed"]
p_iters = params["iters"]

# Загрузка данных
df = pd.read_csv(f_input)

# Определение данных
X = df[["temp", "air_humidity", "pressure"]]
y = df["label"]

# Работа с моделью
model = LogisticRegression(random_state=p_seed, max_iter=p_iters)
model.fit(X, y)

# Сохранение модели
with open(f_output, "wb") as file:
    pickle.dump(model, file)