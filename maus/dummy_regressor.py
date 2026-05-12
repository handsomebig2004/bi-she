from sklearn import dummy
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.signal import resample
import os

x_ecg = []
x_gsr = []
x_inf_ppg = []
x_pix_ppg = []
y = []


for folder_name in os.walk("data/MAUS/Data/Raw_data/"):
    if folder_name[0][-1] != '/':
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ecg.csv").to_numpy().transpose():
            x_ecg.append(list(trial.astype(np.float32)))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_gsr.csv").to_numpy().transpose():
            x_gsr.append(list(trial.astype(np.float32)))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ppg.csv").to_numpy().transpose():
            x_inf_ppg.append(list(trial.astype(np.float32)))
        for trial in  pd.read_csv(f"{folder_name[0]}/pixart.csv").to_numpy().transpose():
            x_pix_ppg.append(list(trial.astype(np.float32)))
        for trial in pd.read_csv(f"data/MAUS/Subjective_rating/{folder_name[0][-3:]}/NASA_TLX.csv").iloc[7, 1:7].to_numpy():
            y.append(np.float32(trial)/100)

# resample data to 4Hz on 30 seconds
resample_size = 120
x_ecg_res = [resample(x, resample_size) for x in x_ecg]
x_gsr_res = [resample(x, resample_size) for x in x_gsr]
x_inf_ppg_res = [resample(x, resample_size) for x in x_inf_ppg]

x_ecg_res_norm = (np.array(x_ecg_res) - np.mean(x_ecg_res)) / np.std(x_ecg_res)
x_gsr_res_norm = (np.array(x_gsr_res) - np.mean(x_gsr_res)) / np.std(x_gsr_res)
x_inf_ppg_res_norm = (np.array(x_inf_ppg_res) - np.mean(x_inf_ppg_res)) / np.std(x_inf_ppg_res)

X = []
y_final = []


for i in range(len(x_ecg_res)):
    X.append(np.concatenate([x_ecg_res_norm[i], x_gsr_res_norm[i], x_inf_ppg_res_norm[i]]))
    y_final.append(y[i])  

X = np.array(X, dtype=np.float32)
y_final = np.array(y_final, dtype=np.float32)
y_final_norm = (y_final - np.mean(y_final)) / np.std(y_final)   

X_train, X_test, y_train, y_test = train_test_split(X, y_final_norm, train_size=0.8, random_state=42)

dummyregressor = dummy.DummyRegressor(strategy='mean')
dummyregressor.fit(X_train, y_train)

print("Score sur test :", dummyregressor.score(X_test, y_test))

