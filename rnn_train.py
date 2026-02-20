import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from rnn import MultiSignalRNN, train_rnn


class RnnDataset(torch.utils.data.Dataset):
    """
    Each item:
      x_tuple = (ecg, gsr, inf_ppg)   each is 1D tensor [L]
      y       = 6D tensor [6]
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    # -----------------------------
    # Config (match FCN pipeline)
    # -----------------------------
    raw_root = "data/MAUS/Data/Raw_data/"
    rating_root = "data/MAUS/Subjective_rating/"
    slice_len = 7_680  # 256Hz * 30s
    duplicate_slices_per_trial = 9  # keep same as your fcn_main.py
    batch_size = 12
    random_state = 42
    n_epochs = 20
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -----------------------------
    # Load + slice into 30s windows
    # -----------------------------
    x_ecg, x_gsr, x_inf_ppg, y = [], [], [], []

    for folder_name in os.walk(raw_root):
        # folder_name[0] is the current directory path
        if folder_name[0][-1] != '/':
            ecg_path = f"{folder_name[0]}/inf_ecg.csv"
            gsr_path = f"{folder_name[0]}/inf_gsr.csv"
            ppg_path = f"{folder_name[0]}/inf_ppg.csv"

            if not (os.path.exists(ecg_path) and os.path.exists(gsr_path) and os.path.exists(ppg_path)):
                continue

            # slice signals (each csv transposed => each "trial" is one 1D array)
            for trial in pd.read_csv(ecg_path).to_numpy().transpose():
                for k in range(len(trial) // slice_len - 1):
                    x_ecg.append(trial[k * slice_len:(k + 1) * slice_len].astype(np.float32))

            for trial in pd.read_csv(gsr_path).to_numpy().transpose():
                for k in range(len(trial) // slice_len - 1):
                    x_gsr.append(trial[k * slice_len:(k + 1) * slice_len].astype(np.float32))

            for trial in pd.read_csv(ppg_path).to_numpy().transpose():
                for k in range(len(trial) // slice_len - 1):
                    x_inf_ppg.append(trial[k * slice_len:(k + 1) * slice_len].astype(np.float32))

            # labels: 6D TLX vector, duplicated to match slices
            subj_id = folder_name[0][-3:]
            tlx_path = f"{rating_root}/{subj_id}/NASA_TLX.csv"
            if os.path.exists(tlx_path):
                # same as your fcn_main.py: iloc[0:6, 1:7].to_numpy().transpose()
                # -> each 'trial' is a length-6 vector
                
                tlx_trials = pd.read_csv(tlx_path).iloc[0:6, 1:7].to_numpy().transpose().astype(np.float32)

                for tlx_vec in tlx_trials:
                    for _ in range(duplicate_slices_per_trial):
                        y.append(tlx_vec)

    print("Loaded slices:", len(x_ecg), len(x_gsr), len(x_inf_ppg))
    print("Loaded labels:", len(y))
    if len(y) > 0:
        print("Example y[0]:", y[0])

    # Basic sanity check
    n = min(len(x_ecg), len(x_gsr), len(x_inf_ppg), len(y))
    x_ecg, x_gsr, x_inf_ppg, y = x_ecg[:n], x_gsr[:n], x_inf_ppg[:n], y[:n]
    print("After align:", len(x_ecg), len(y))

    # -----------------------------
    # To torch + normalize (match FCN)
    # -----------------------------
    x_ecg = torch.nn.functional.normalize(torch.tensor(np.stack(x_ecg), dtype=torch.float32))
    x_gsr = torch.nn.functional.normalize(torch.tensor(np.stack(x_gsr), dtype=torch.float32))
    x_inf_ppg = torch.nn.functional.normalize(torch.tensor(np.stack(x_inf_ppg), dtype=torch.float32))
    y = torch.tensor(np.stack(y), dtype=torch.float32)  # [N,6]

    # -----------------------------
    # Train/test split (match FCN)
    # -----------------------------
    x_ecg_train, x_ecg_test, x_gsr_train, x_gsr_test, x_inf_train, x_inf_test, y_train, y_test = train_test_split(
        x_ecg, x_gsr, x_inf_ppg, y,
        train_size=0.8,
        random_state=random_state
    )

    train_dataset = RnnDataset(list(zip(x_ecg_train, x_gsr_train, x_inf_train)), y_train)
    test_dataset  = RnnDataset(list(zip(x_ecg_test,  x_gsr_test,  x_inf_test)),  y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)

    # -----------------------------
    # Model / loss / optimizer
    # -----------------------------
    rnn_net = MultiSignalRNN(num_signals=3, hidden_size=128, num_layers=1, dropout=0.0, out_dim=6)
    loss_func = torch.nn.MSELoss()
    optim_adam = torch.optim.Adam(params=rnn_net.parameters(), lr=lr)

    # -----------------------------
    # Train
    # -----------------------------
    t0 = time.time()
    train_loss_list, valid_loss_list = train_rnn(
        rnn_net,
        train_loader,
        test_loader,
        loss_func,
        optim_adam,
        n_epochs=n_epochs,
        device=device,
        print_samples=True
    )
    t1 = time.time()
    print("Total time (s):", t1 - t0)

    # -----------------------------
    # Plot losses
    # -----------------------------
    plt.plot(range(len(train_loss_list)), train_loss_list, label="train")
    plt.plot(range(len(valid_loss_list)), valid_loss_list, label="valid")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
