import sys
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


if len(sys.argv) != 2:
    sys.stderr.write("Ошибка в аргументах. Пример выполнения:\n")
    sys.stderr.write("python get_features.py data_file\n")
    sys.exit(1)

f_input = sys.argv[1]
f_input_train = f"{f_input}/train.csv"
f_input_test = f"{f_input}/test.csv"
f_output_train = os.path.join("data", "prepared", "train.csv")
f_output_test = os.path.join("data", "prepared", "test.csv")
os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

# Чтение необработанных данных
df_train = pd.read_csv(f_input_train)
df_test = pd.read_csv(f_input_test)

# Обработка вероятных пропусков
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

# Работа с категориальными переменными
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

# Выбор функции для предобработки
corr_matrix = df_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k = 1).astype(bool))

to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
df_train.drop(to_drop, axis=1, inplace=True)
df_test.drop(to_drop, axis=1, inplace=True)


scaler = StandardScaler()
df_train[["temp", "air_humidity", "pressure"]] = scaler.fit_transform(
    df_train[["temp", "air_humidity", "pressure"]])
df_test[["temp", "air_humidity", "pressure"]] = scaler.transform(
    df_test[["temp", "air_humidity", "pressure"]])

# Сохранение предобработанных данных
df_train.to_csv(f_output_train, index=False)
df_test.to_csv(f_output_test, index=False)