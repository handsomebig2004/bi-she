
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from load_data import train_res_data_loader, valid_res_data_loader, test_res_data_loader, train_res_data_loader_list, test_res_data_loader_list

class LSTM(nn.Module):

    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 384, batch_first=True)
        self.swish = nn.SiLU()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(384, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        
        x = self.conv(x)          # (batch, 64, 120)
        x = self.relu(x)

        x = x.permute(0, 2, 1)    # (batch, 120, 64)
        x, _ = self.lstm(x)       # (batch, 120, 384)

        x = x[:, -1, :]           # (batch, 384) -> dernier timestep

        x = self.swish(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.linear1(x)       # (batch, 1)


        return x


def valid_epoch(test_loader, loss_func, model):
    
    model.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for x_batche_l, y_batch in test_loader:
            
            preds = model(x_batche_l)
            
            loss = loss_func(preds.squeeze(), y_batch)
            
            n_samples += y_batch.size(0)
            tot_loss += loss.item() * y_batch.size(0)

    model.train()
    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    return avg_loss


def train_LSTM(train_res_data_loader, valid_res_data_loader, optimizer, loss_fn, n_epochs, model, verb=1):

    train_loss_list = []
    valid_loss_list = []
    
    epoch_loss = 0
    n_samples = 0

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_res_data_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs.squeeze(), y_batch)
            epoch_loss += loss.item() * y_batch.size(0)
            n_samples += y_batch.size(0)

            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            valid_loss = valid_epoch(valid_res_data_loader, loss_fn, model)
            valid_loss_list.append(valid_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {(epoch_loss/n_samples):.4f}, Valid loss; {valid_loss:.4f}")
        train_loss_list.append(epoch_loss/n_samples)
        
    if verb:
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train')
        plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid')
        
        #print(f'test mse: {valid_epoch(test_res_data_loader, loss_fn, model)}')
        #print(f'test mae: {valid_epoch(test_res_data_loader, nn.L1Loss(), model)}')

        plt.legend()
        plt.show()
    
    return valid_loss
    
def train_lopo_LSTM(train_res_data_loader_list, valid_res_data_loader_list, optimizer, loss_fn, n_epochs, verb=1):
    if len(train_res_data_loader_list) != len(valid_res_data_loader_list):
        raise TabError(f"different size for train_res_data_loader_list and valid_res_data_loader_list\n\
                        got {len(train_res_data_loader_list)} and {len(valid_res_data_loader_list)}")
    else:
        model = LSTM(input_size=3)
        valid_loss_list = []
        for i in range(len(train_res_data_loader_list)):
            train_data_loader = train_res_data_loader_list[i]
            test_data_loader = test_res_data_loader_list[i]
            valid_loss = train_LSTM(train_res_data_loader=train_data_loader,
                                    valid_res_data_loader=test_data_loader,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    n_epochs=n_epochs,
                                    model=model,
                                    verb=verb)
            valid_loss_list.append(valid_loss)
            print(f'valid_loss: {valid_loss}')
        plt.scatter(list(range(len(train_res_data_loader_list))), valid_loss_list)
        plt.show()

if __name__ == "__main__":
    
    model = LSTM(input_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    n_epochs = 20    
    
    train_lopo_LSTM(train_res_data_loader_list=train_res_data_loader_list,
                    valid_res_data_loader_list=test_res_data_loader_list,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    n_epochs=2,
                    verb=0)
    
    train_LSTM(train_res_data_loader=train_res_data_loader,
               valid_res_data_loader=valid_res_data_loader,
               optimizer=optimizer,
               loss_fn=loss_fn,
               n_epochs=n_epochs,
               model=model)
    
    