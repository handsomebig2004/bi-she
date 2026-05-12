import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


def normalise_data():
    min_max_scaler = MinMaxScaler()
    path = 'data/MAUS/Data/Raw_data'
    normalized_data_path = 'data/MAUS/Data/Normalized_data'
    for folder_name in os.listdir(path):
        for file in os.listdir(path + '/' + folder_name):
            try:
                open(f'{normalized_data_path}/{folder_name}_normalized_{file}')
            except:
                if file[-3:] == 'csv':
                    df = pd.read_csv(f'{path}/{folder_name}/{file}')
                else:
                    df = pd.read_excel(f'{path}/{folder_name}/{file}')
                df = pd.DataFrame(min_max_scaler.fit_transform(df))
                df.to_csv(f'{normalized_data_path}/{folder_name}_normalized_{file}')
            
        

if __name__ == "__main__":
    
    normalise_data()