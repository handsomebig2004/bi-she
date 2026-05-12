import torch
import torch.nn as nn
import time as t
import matplotlib.pyplot as plt
import numpy as np

class FCNBranch(nn.Module):
    def __init__(self, kernel_size):
        super(FCNBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size) #shape batch_size*16*L_out_1 
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size) #shape batch_size*32*L_out_2
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size) #shape batch_size*64*L_out_3
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1) # Averages all 64 feature maps to get an output of shape batch_size*64*1

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.gap(x)
        return x

class FCNModel(nn.Module):
    def __init__(self, num_signals, kernel_size):
        super().__init__()
        self.kernel_size=kernel_size
        self.num_signals=num_signals

        # separates all signals in separate branches 
        self.branches = nn.ModuleList([FCNBranch(kernel_size) for _ in range(num_signals)])

        #regression with a fully connected layer
        self.fc = nn.Linear(in_features=64*num_signals, out_features=1)

    def forward(self, x_list):
        # x_list : liste de tenseurs [batch, 1, L] pour chaque signal
        outputs = []
        for i, branch in enumerate(self.branches):
            out = branch(x_list[i])
            outputs.append(out)
        # Concatenation
        x = torch.cat(outputs, dim=1)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


def _prepare_tensor(x):
    # x may be tensor, numpy array, list or (tensor,) from dataloader collate
    if isinstance(x, (list, tuple)):
        x = x[0]
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 2:  # [batch, L] -> add channel dim
        x = x.unsqueeze(1)
    return x

def _prepare_label(y):
        # x may be tensor, numpy array, list or (tensor,) from dataloader collate
    if isinstance(y, (list, tuple)):
        y = y[0]
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    return y



def epoch_train(_net, train_loader, loss_func, optim):
    """An epoch of training for this model

    Parameters
    ----------
    _net : FCNModel()
        the fcn model to train
    train_loader_l : list[torch.utils.data.DataLoader()]
        A list of training input dataloaders of length the number of signals
    y_train_loader : torch.utils.data.DataLoader()
        A training output dataloader containing float32 
    loss_func : func
        The loss function to be used
    optim : torch.optim
        the optimizer to use

    Returns
    -------
    int
        the average training loss over all elements for this epoch
    """
    _net.train()
    tot_loss, n_samples=0,0

    for x_batche_l, y_batch in train_loader:
        optim.zero_grad()
        #preparing all inputs
        x_list = [_prepare_tensor(xb) for xb in x_batche_l]
        y = _prepare_label(y_batch)

        #prediction and step
        preds=_net(x_list)

        loss=loss_func(preds.squeeze(), y.float())

        loss.backward()
        optim.step()

        #computing training loss
        _net.eval()
        n_samples += y.size(0)
        tot_loss += loss.item() * y.size(0)
        _net.train()
    
    avg_loss = tot_loss / n_samples
    return avg_loss


def epoch_valid(_net, valid_loader, loss_func):
    """Tests network performance with a validation dataset 

    Parameters
    ----------
    _net : FCNModel()
        the fcn model to test
    valid_loader : list[torch.utils.data.DataLoader()]
        input dataloader
    loss_func : func
        The loss function to be used

    Returns
    -------
    int
        the average loss over all elements for this epoch
    """
    _net.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for x_batche_l, y_batch in valid_loader:
            #preparing all inputs

            x_list = [_prepare_tensor(xb) for xb in x_batche_l]
            y = _prepare_label(y_batch)

            preds = _net(x_list)
            if tot_loss == 0:
                print(preds[:5])
                print(y[:5])
                # plt.plot(range(len(preds)), preds)
                # plt.plot(range(len(y)), y)
                # plt.show()

            loss = loss_func(preds.squeeze(), y.float())

            n_samples += y.size(0)
            tot_loss += loss.item() * y.size(0)

    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    return avg_loss

def train(_net, train_loader, valid_loader, loss_func, optim, n_epochs):
    """A training loop for the provided network for n_epochs

    Parameters
    ----------
    _net : FCNModel()
        the fcn model to train
    train_loader : list[torch.utils.data.DataLoader()]
        input dataloader
    valid_loader : torch.utils.data.DataLoader()
        validation dataloader
    loss_func : func
        The loss function to be used
    optim : torch.optim
        the optimizer to use
    n_epochs : int
        the number of epochs to train for

    Returns
    -------
    tuple(list[int], list[int])
        a tuple containg the lists of training and validation losses over the epochs
    """
    train_loss_list, valid_loss_list=[],[]
    _net.train()

    for epoch in range(n_epochs):
        t_start=t.time()

        #training loop
        train_loss=epoch_train(_net, train_loader, loss_func, optim)
        train_loss_list.append(train_loss)

        with torch.no_grad():
            #valid ation
            valid_loss = epoch_valid(_net, valid_loader, loss_func)
            _net.train()
            valid_loss_list.append(valid_loss)
        t_end=t.time()
        print(f'Epoch {epoch}:  train loss {train_loss}, valid loss {valid_loss}')
        print(f'Temps écoulé: {t_end-t_start}')
    return train_loss_list, valid_loss_list


def test_model(_net, test_loader, loss_func):
    _net.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for x_batche_l, y_batch in test_loader:
            #preparing all inputs

            x_list = [_prepare_tensor(xb) for xb in x_batche_l]
            y = _prepare_label(y_batch)

            preds = _net(x_list)
            if tot_loss == 0:
                print(preds[:5])
                print(y[:5])
                # plt.plot(range(len(preds)), preds)
                # plt.plot(range(len(y)), y)
                # plt.show()

            loss = loss_func(preds.squeeze(), y.float())

            n_samples += y.size(0)
            tot_loss += loss.item() * y.size(0)

    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    return avg_loss