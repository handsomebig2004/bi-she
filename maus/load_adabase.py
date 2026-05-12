import pandas as pd
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from scipy.signal import spectrogram

NUM_PATIENTS = 30

SKT_SR = 100
ECG_SR = 500
RSP_SR = 250
EMG_SR = 1000
EDA_SR = 500
EYE_SR = 250

data_folder="data/adabase"

#studied signal column names
signal_columns=['RAW_ECG_I',
'RAW_ECG_II',
'RAW_SKT',
'RAW_RSP',
'RAW_EMG',
'RAW_EDA',
'LEFT_GAZE_POINT_VALIDITY',
'LEFT_PUPIL_DIAMETER',
'LEFT_PUPIL_VALIDITY',
'LEFT_GAZE_ORIGIN_VALIDITY',
'RIGHT_GAZE_POINT_VALIDITY',
'RIGHT_PUPIL_DIAMETER',
'RIGHT_PUPIL_VALIDITY',
'RIGHT_GAZE_ORIGIN_VALIDITY',
'LEFT_GAZE_POINT_ON_DISPLAY_AREA_X',
'LEFT_GAZE_POINT_ON_DISPLAY_AREA_Y',
'RIGHT_GAZE_POINT_ON_DISPLAY_AREA_X',
'RIGHT_GAZE_POINT_ON_DISPLAY_AREA_Y',
'LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM_X',
'LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM_Y',
'LEFT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM_Z',
'RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM_X',
'RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM_Y',
'RIGHT_GAZE_POINT_IN_USER_COORDINATE_SYSTEM_Z',
'LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM_X',
'LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM_Y',
'LEFT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM_Z',
'RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM_X',
'RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM_Y',
'RIGHT_GAZE_ORIGIN_IN_USER_COORDINATE_SYSTEM_Z',
'LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM_X',
'LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM_Y',
'LEFT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM_Z',
'RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM_X',
'RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM_Y',
'RIGHT_GAZE_ORIGIN_IN_TRACKBOX_COORDINATE_SYSTEM_Z',]
eye_columns=signal_columns[6: -1]


tlx_columns=['EFFORT',
'FRUSTRATION',
'MENTAL',
'PERFORMANCE',
'PHYSICAL',
'TEMPORAL',
'WEIGHT EFFORT',
'WEIGHT FRUSTRATION',
'WEIGHT MENTAL',
'WEIGHT PERFORMANCE',
'WEIGHT PHYSICAL',
'WEIGHT TEMPORAL']

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

def split_data(x_list, y_subcategories, y_weights, train_indices, valid_indices, test_indices):
    train_list = []
    valid_list = []
    test_list = []
    for x in x_list:
        train_list.append(torch.tensor([x[i] for i in train_indices]))
        valid_list.append(torch.tensor([x[i] for i in valid_indices]))
        test_list.append(torch.tensor([x[i] for i in test_indices]))
    y_subcategories_train, y_subcategories_valid, y_subcategories_test = [y_subcategories[i] for i in train_indices], [y_subcategories[i] for i in valid_indices], [y_subcategories[i] for i in test_indices]
    y_weights_train, y_weights_valid, y_weights_test = [y_weights[i] for i in train_indices], [y_weights[i] for i in valid_indices], [y_weights[i] for i in test_indices]
    return train_list, valid_list, test_list, y_subcategories_train, y_subcategories_valid, y_subcategories_test, y_weights_train, y_weights_valid, y_weights_test



############################################
########## fetch data from files ###########
############################################

file_names = os.listdir(data_folder)
file_names = list(filter(lambda x: x.endswith(".h5py"), file_names))


x_list=[[]for _ in range(len(signal_columns))]
x_list_res=[[]for _ in range(len(signal_columns))]
y_subcategories=[]
y_weights=[]



print('loading files')
for patient_number in range(NUM_PATIENTS):

    file_name = file_names[patient_number]
    file_path = os.path.join(data_folder , file_name)   

    # Load data
    df_signals = pd.read_hdf(file_path, "SIGNALS", mode="r")
    df_subjective = pd.read_hdf(file_path, "SUBJECTIVE", mode="r")

    frag_length = 30
    print(f'loaded patient n° {patient_number}')
    

    #number of inputs (adabase is indexed by a timestamp in milliseconds)
    n_input=1000*30

    df_signals[eye_columns] = df_signals[eye_columns].replace([np.inf, -np.inf], np.nan)

    df_signals_resampled=df_signals.dropna(axis=0)

    y_done=False

    for i,signal in enumerate(signal_columns):  
        signal_list=[]
        signal_list_resampled=[]
        
        n_events=len(df_signals)


        current_idx=0

        current_val=df_signals[['STUDY', 'PHASE', 'LEVEL']].iloc[current_idx]
        mask=(df_signals[['STUDY', 'PHASE', 'LEVEL']].iloc[current_idx:] != current_val).any(axis=1)
        next_trial_change=mask.argmax() if mask.argmax()!=current_idx else np.inf
        end_idx=min(0+n_input, n_events, next_trial_change)

        while current_idx<n_input-1:

            cleaned_segment=df_signals[signal].dropna().iloc[current_idx:end_idx]
            signal_list.append(cleaned_segment.astype(np.float32).to_numpy().transpose())

            signal_list_resampled.append(df_signals_resampled[signal].dropna().iloc[current_idx:end_idx].astype(np.float32).to_numpy().transpose())

            current_trial=df_signals[['STUDY', 'PHASE', 'LEVEL']].iloc[int(current_idx)]


            if not y_done:

                y_subcategories.append(df_subjective[['EFFORT',
                                                        'FRUSTRATION',
                                                        'MENTAL',
                                                        'PERFORMANCE',
                                                        'PHYSICAL',
                                                        'TEMPORAL']].loc[(current_trial['STUDY'] == df_subjective['STUDY']) &
                                                                         (current_trial['PHASE'] == df_subjective['PHASE']) &
                                                                         (current_trial['LEVEL'] == df_subjective['LEVEL'])].astype(np.float32).to_numpy().transpose())
                y_weights.append(df_subjective[['WEIGHT EFFORT',
                                                'WEIGHT FRUSTRATION',
                                                'WEIGHT MENTAL',
                                                'WEIGHT PERFORMANCE',
                                                'WEIGHT PHYSICAL',
                                                'WEIGHT TEMPORAL']].loc[(current_trial['STUDY'] == df_subjective['STUDY']) &
                                                                        (current_trial['PHASE'] == df_subjective['PHASE']) &
                                                                        (current_trial['LEVEL'] == df_subjective['LEVEL'])].astype(np.float32).to_numpy().transpose())

                y_done=True

            mask=(df_signals[['STUDY', 'PHASE', 'LEVEL']].iloc[current_idx:] != current_trial).any(axis=1)
            next_trial_change=mask.argmax() if id!=current_idx else np.inf


            current_idx, end_idx=end_idx, min(current_idx+n_input, n_events, next_trial_change)

    
        x_list[i].append(signal_list.copy())
        x_list_res[i].append(signal_list_resampled.copy())
print('loaded and formatted all files')

x_signal_l_norm=[]
for x in x_list:
    x_signal_l_norm = torch.nn.functional.normalize(torch.tensor(x))


indices = list(range(NUM_PATIENTS))

train_indices_base, test_indices_base = train_test_split(indices, test_size=0.1)
train_indices_base, valid_indices_base = train_test_split(train_indices_base, test_size=0.2)

train_indices = [x * (len(x_list[0]) // NUM_PATIENTS) + i for i in range(len(x_list[0]) // NUM_PATIENTS) for x in train_indices_base]
valid_indices = [x * (len(x_list[0]) // NUM_PATIENTS) + i for i in range(len(x_list[0]) // NUM_PATIENTS) for x in valid_indices_base]
test_indices = [x * (len(x_list[0]) // NUM_PATIENTS) + i for i in range(len(x_list[0]) // NUM_PATIENTS) for x in test_indices_base]

train_indices_res = [x * (len(x_list_res[0]) // NUM_PATIENTS) + i for i in range(len(x_list_res[0]) // NUM_PATIENTS) for x in train_indices_base]
valid_indices_res = [x * (len(x_list_res[0]) // NUM_PATIENTS) + i for i in range(len(x_list_res[0]) // NUM_PATIENTS) for x in valid_indices_base]
test_indices_res = [x * (len(x_list_res[0]) // NUM_PATIENTS) + i for i in range(len(x_list_res[0]) // NUM_PATIENTS) for x in test_indices_base]


x_all=[]
for i in range(len(x_list[0])):
    signals=np.stack([s[i] for s in x_list], axis=0)
    x_all.append(signals)

x_all_res=[]
for i in range(len(x_list_res[0])):
    signals=np.stack([s[i] for s in x_list_res], axis=0)
    x_all_res.append(signals)

######################################
########## resampled data ############
######################################


x_train_res_list, x_valid_res_list, x_test_res_list, y_res_subcategories_train, y_res_subcategories_valid, y_res_subcategories_test, y_res_weights_train, y_res_weights_valid, y_res_weights_test = split_data([x_all], y_subcategories, y_weights, train_indices_res, valid_indices_res, test_indices_res)

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
x_train_res_list, x_valid_res_list, x_test_res_list, y_freq_train, y_freq_valid, y_freq_test = split_data([final_signal], y_sub_categories, train_indices, valid_indices, test_indices)

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


