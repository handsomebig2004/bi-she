import pandas as pd
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from scipy.signal import spectrogram

NUM_PATIENTS = 22

class FcnDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def split_data(x_list, y, train_indices, valid_indices, test_indices):
    train_list = []
    valid_list = []
    test_list = []
    for x in x_list:
        train_list.append(torch.tensor([x[i] for i in train_indices]))
        valid_list.append(torch.tensor([x[i] for i in valid_indices]))
        test_list.append(torch.tensor([x[i] for i in test_indices]))
    y_train, y_valid, y_test = [y[i] for i in train_indices], [y[i] for i in valid_indices], [y[i] for i in test_indices]
    return train_list, valid_list, test_list, y_train, y_valid, y_test



############################################
########## fetch data from files ###########
############################################

x_ecg = []
x_gsr = []
x_inf_ppg = []
x_pix_ppg = []
y = []
y_sub_categories = []
y_weight = []

for folder_name in os.walk("data/MAUS/Data/Raw_data/"):
    if folder_name[0][-1] != '/':
        
        frag_length = 30
        
        # 256 Hz
        n_input_256 = 256 * frag_length
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ecg.csv").to_numpy().transpose():
            for k in range(len(trial) // n_input_256 - 1):
                x_ecg.append(list(trial[k*n_input_256:(k+1)*n_input_256].astype(np.float32)))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_gsr.csv").to_numpy().transpose():
            for k in range(len(trial) // n_input_256 - 1):
                x_gsr.append(list(trial[k*n_input_256:(k+1)*n_input_256].astype(np.float32)))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ppg.csv").to_numpy().transpose():
            for k in range(len(trial) // n_input_256 - 1):
                x_inf_ppg.append(list(trial[k*n_input_256:(k+1)*n_input_256].astype(np.float32)))
            
        # 100 Hz
        n_input_100 = 100 * frag_length
        for trial in  pd.read_csv(f"{folder_name[0]}/pixart.csv").to_numpy().transpose():
            #print(trial.shape)
            for k in range(len(trial) // n_input_100 - 1):
                x_pix_ppg.append(list(trial[k*n_input_100:(k+1)*n_input_100].astype(np.float32)))


for folder_name in os.walk("data/MAUS/Data/Raw_data/"):
    if folder_name[0][-1] != '/':
        for trial in pd.read_csv(f"data/MAUS/Subjective_rating/{folder_name[0][-3:]}/NASA_TLX.csv").iloc[7, 1:7].to_numpy().transpose():
            for k in range(int(len(x_ecg) / 132)): # duplicate results for the same trial (since we split the in 30s slices)
                y.append(np.float32(trial))
        for trial in pd.read_csv(f"data/MAUS/Subjective_rating/{folder_name[0][-3:]}/NASA_TLX.csv").iloc[0:6, 1:7].to_numpy().transpose():
            for k in range(int(len(x_ecg) / 132)): # duplicate results for the same trial (since we split the in 30s slices)
                y_sub_categories.append(np.float32(trial))
        for k in range(int(len(x_ecg) / 132)): # duplicate results for the same trial (since we split the in 30s slices)
            y_weight.append(pd.read_csv(f"data/MAUS/Subjective_rating/{folder_name[0][-3:]}/NASA_TLX.csv").iloc[0:6, 7].to_numpy())

x_inf_ppg_norm = torch.nn.functional.normalize(torch.tensor(x_inf_ppg))
x_ecg_norm = torch.nn.functional.normalize(torch.tensor(x_ecg))
x_gsr_norm = torch.nn.functional.normalize(torch.tensor(x_gsr))
x_pix_ppg_norm = torch.nn.functional.normalize(torch.tensor(x_pix_ppg))

indices = list(range(NUM_PATIENTS))

train_indices_base, test_indices_base = train_test_split(indices, test_size=0.1)
train_indices_base, valid_indices_base = train_test_split(train_indices_base, test_size=0.2)

train_indices = [x * (len(x_ecg) // NUM_PATIENTS) + i for i in range(len(x_ecg) // NUM_PATIENTS) for x in train_indices_base]
valid_indices = [x * (len(x_ecg) // NUM_PATIENTS) + i for i in range(len(x_ecg) // NUM_PATIENTS) for x in valid_indices_base]
test_indices = [x * (len(x_ecg) // NUM_PATIENTS) + i for i in range(len(x_ecg) // NUM_PATIENTS) for x in test_indices_base]


##############################################################
####### Leave one person out for pretrained and LSTM #########
##############################################################

train_freq_data_loader_list = []
test_freq_data_loader_list = []
train_res_data_loader_list = []
test_res_data_loader_list = []

for patient_index in range(NUM_PATIENTS):
    train_indices_lopo = list(range(NUM_PATIENTS))
    train_indices_lopo.remove(patient_index)
    
    # res data
    resample_size = 120
    x_ecg_res = [resample(x, resample_size) for x in x_ecg]
    x_gsr_res = [resample(x, resample_size) for x in x_gsr]
    x_inf_ppg_res = [resample(x, resample_size) for x in x_inf_ppg]
    x_pix_ppg_res = [resample(x, resample_size) for x in x_pix_ppg]

    train_indices_res = [x * (len(x_ecg_res) // NUM_PATIENTS) + i for i in range(len(x_ecg_res) // NUM_PATIENTS) for x in train_indices_lopo]
    test_indices_res = [patient_index * (len(x_ecg_res) // NUM_PATIENTS) + i for i in range(len(x_ecg_res) // NUM_PATIENTS)]

    x_all = []
    for i in range(len(x_ecg_res)):
        signals = np.stack([x_ecg_res[i], x_gsr_res[i], x_inf_ppg_res[i]], axis=0)
        x_all.append(signals)

    x_train_res_list, _, x_test_res_list, y_res_train, _, y_res_test = split_data([x_all], y_sub_categories, train_indices_res, [], test_indices_res)

    _train_res_data_loader = torch.utils.data.DataLoader(list(zip(x_train_res_list[0], y_res_train)), batch_size=32, shuffle=False)
    _test_res_data_loader = torch.utils.data.DataLoader(list(zip(x_test_res_list[0], y_res_test)), batch_size=32, shuffle=False)
    
    train_res_data_loader_list.append(_train_res_data_loader)
    test_res_data_loader_list.append(_test_res_data_loader)

    #pour x_ecg_res : 
    for i in range (len(x_ecg_res)) :
        fs = 4
        t = np.linspace(0,30,fs*30)
        signal = x_ecg_res[i]
        f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
        x_ecg_res[i] = Sxx

    #pour x_gsr_res : 
    for i in range (len(x_gsr_res)) :
        fs = 4
        t = np.linspace(0,30,fs*30)
        signal = x_gsr_res[i]
        f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
        x_gsr_res[i] = Sxx

    #pour x_inf_ppg_res : 
    for i in range (len(x_inf_ppg_res)) : 
        fs = 4
        t = np.linspace(0,30,fs*30)
        signal = x_inf_ppg_res[i]
        f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
        x_inf_ppg_res[i] = Sxx

    #pour x_pix_ppg_res : 
    for i in range (len(x_pix_ppg_res)) : 
        fs = 4
        t = np.linspace(0,30,fs*30)
        signal = x_pix_ppg_res[i]
        f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
        x_pix_ppg_res[i] = Sxx
        
    final_signal = np.stack([x_ecg_res, x_gsr_res, x_inf_ppg_res, x_pix_ppg_res], axis=0)   
    final_signal = final_signal.transpose(1,0,2,3)

    # resampled data loaders
    x_train_res_list, _, x_test_res_list, y_freq_train, _, y_freq_test = split_data([final_signal], y_sub_categories, train_indices, [], test_indices)

    train_freq_data_loader_list.append(torch.utils.data.DataLoader(list(zip(x_train_res_list[0], y_freq_train, y_weight)), batch_size=32, shuffle=False))
    test_freq_data_loader_list.append(torch.utils.data.DataLoader(list(zip(x_test_res_list[0], y_freq_test, y_weight)), batch_size=32, shuffle=False))


######################################
########## resampled data ############
######################################

resample_size = 120
x_ecg_res = [resample(x, resample_size) for x in x_ecg]
x_gsr_res = [resample(x, resample_size) for x in x_gsr]
x_inf_ppg_res = [resample(x, resample_size) for x in x_inf_ppg]
x_pix_ppg_res = [resample(x, resample_size) for x in x_pix_ppg]

train_indices_res = [x * (len(x_ecg_res) // NUM_PATIENTS) + i for i in range(len(x_ecg_res) // NUM_PATIENTS) for x in train_indices_base]
valid_indices_res = [x * (len(x_ecg_res) // NUM_PATIENTS) + i for i in range(len(x_ecg_res) // NUM_PATIENTS) for x in valid_indices_base]
test_indices_res = [x * (len(x_ecg_res) // NUM_PATIENTS) + i for i in range(len(x_ecg_res) // NUM_PATIENTS) for x in test_indices_base]

x_all = []
for i in range(len(x_ecg_res)):
    signals = np.stack([x_ecg_res[i], x_gsr_res[i], x_inf_ppg_res[i]], axis=0)
    x_all.append(signals)

x_train_res_list, x_valid_res_list, x_test_res_list, y_res_train, y_res_valid, y_res_test = split_data([x_all], y, train_indices_res, valid_indices_res, test_indices_res)

train_res_data_loader = torch.utils.data.DataLoader(list(zip(x_train_res_list[0], y_res_train)), batch_size=32, shuffle=False)
valid_res_data_loader = torch.utils.data.DataLoader(list(zip(x_valid_res_list[0], y_res_valid)), batch_size=32, shuffle=False)
test_res_data_loader = torch.utils.data.DataLoader(list(zip(x_test_res_list[0], y_res_test)), batch_size=32, shuffle=False)

#pour x_ecg_res : 
for i in range (len(x_ecg_res)) :
    fs = 4
    t = np.linspace(0,30,fs*30)
    signal = x_ecg_res[i]
    f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
    x_ecg_res[i] = Sxx

#pour x_gsr_res : 
for i in range (len(x_gsr_res)) :
    fs = 4
    t = np.linspace(0,30,fs*30)
    signal = x_gsr_res[i]
    f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
    x_gsr_res[i] = Sxx

#pour x_inf_ppg_res : 
for i in range (len(x_inf_ppg_res)) : 
    fs = 4
    t = np.linspace(0,30,fs*30)
    signal = x_inf_ppg_res[i]
    f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
    x_inf_ppg_res[i] = Sxx

#pour x_pix_ppg_res : 
for i in range (len(x_pix_ppg_res)) : 
    fs = 4
    t = np.linspace(0,30,fs*30)
    signal = x_pix_ppg_res[i]
    f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
    x_pix_ppg_res[i] = Sxx
    
final_signal = np.stack([x_ecg_res, x_gsr_res, x_inf_ppg_res, x_pix_ppg_res], axis=0)   
final_signal = final_signal.transpose(1,0,2,3)

# resampled data loaders
x_train_res_list, x_valid_res_list, x_test_res_list, y_freq_train, y_freq_valid, y_freq_test = split_data([final_signal], y, train_indices, valid_indices, test_indices)

train_freq_data_loader = torch.utils.data.DataLoader(list(zip(x_train_res_list[0], y_freq_train, y_weight)), batch_size=32, shuffle=False)
valid_freq_data_loader = torch.utils.data.DataLoader(list(zip(x_valid_res_list[0], y_freq_valid, y_weight)), batch_size=32, shuffle=False)
test_freq_data_loader = torch.utils.data.DataLoader(list(zip(x_test_res_list[0], y_freq_test, y_weight)), batch_size=32, shuffle=False)



######################################################
########### not normalized data loaders ##############
######################################################

x_train_list, x_valid_list, x_test_list, y_train, y_valid, y_test = split_data([x_inf_ppg, x_ecg, x_gsr, x_pix_ppg], y, train_indices, valid_indices, test_indices)

train_dataset = FcnDataset(list(zip(x_train_list[0],
                               x_train_list[1], 
                               x_train_list[2], 
                               x_train_list[3])), y_train)
valid_dataset = FcnDataset(list(zip(x_valid_list[0],
                               x_valid_list[1],
                               x_valid_list[2],
                               x_valid_list[3])), y_valid)
test_dataset = FcnDataset(list(zip(x_test_list[0],
                              x_test_list[1],
                              x_test_list[2],
                              x_test_list[3],)), y_test)

# not normalized data loaders
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=12)
valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset, shuffle=True, batch_size=12)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=12)



######################################
###### normalized data loaders #######
######################################
'''
x_norm_train_list, x_norm_valid_list, x_norm_test_list, y_train, y_valid, y_test = split_data([x_inf_ppg_norm, x_ecg_norm, x_gsr_norm, x_pix_ppg_norm], y, train_indices, valid_indices, test_indices)

train_norm_dataset = FcnDataset(list(zip(x_train_list[0],
                               x_train_list[1], 
                               x_train_list[2], 
                               x_train_list[3])), y_train)
valid_norm_dataset = FcnDataset(list(zip(x_valid_list[0],
                               x_valid_list[1],
                               x_valid_list[2],
                               x_valid_list[3])), y_valid)
test_norm_dataset = FcnDataset(list(zip(x_test_list[0],
                              x_test_list[1],
                              x_test_list[2],
                              x_test_list[3],)), y_test)
                              
# normalized data loaders
train_data_norm_loader = torch.utils.data.DataLoader(dataset=train_norm_dataset, shuffle=True, batch_size=12)
valid_data_norm_loader = torch.utils.data.DataLoader(dataset=valid_norm_dataset, shuffle=True, batch_size=12)
test_data_norm_loader = torch.utils.data.DataLoader(dataset=test_norm_dataset, shuffle=True, batch_size=12)
                              '''


