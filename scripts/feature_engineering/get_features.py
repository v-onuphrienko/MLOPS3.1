import sys
import os

import pandas as pd

if len(sys.argv) != 2:
    sys.stderr.write("Ошибка в аргументах. Пример выполнения:\n")
    sys.stderr.write("python get_features.py data_file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_input_train = f"{f_input}/train.csv"
f_input_test = f"{f_input}/test.csv"
f_output_train = os.path.join("data", "features", "train.csv")
f_output_test = os.path.join("data", "features", "test.csv")
os.makedirs(os.path.join("data", "features"), exist_ok=True)

df_train = pd.read_csv(f_input_train)
df_test = pd.read_csv(f_input_test)

df_train.to_csv(f_output_train, index=False)
df_test.to_csv(f_output_test, index=False)