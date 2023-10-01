import os
import sys
import pickle
import json

import pandas as pd

if len(sys.argv) != 3:
    sys.stderr.write("Ошибка в аргументах. Пример выполнения:\n")
    sys.stderr.write("\tpython evaluate.py data-file model\n")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
X = df[["temp", "air_humidity", "pressure"]]
y = df["label"]

with open(sys.argv[2], "rb") as fd:
    model = pickle.load(fd)

score = model.score(X, y)

prc_file = "evaluate.json"

with open(prc_file, "w") as fd:
    json.dump({"score": score}, fd)