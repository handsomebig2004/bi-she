import torch
import torch.nn as nn
from torchvision import models
from load_data import train_freq_data_loader, valid_freq_data_loader, test_freq_data_loader
import matplotlib.pyplot as plt

models_list = []
optimizer_list = []
loss_list = []

def feature_extraction(y_batch,i) :
    res = []
    for sample in y_batch : 
        res.append(sample[i])
    return res

for i in range(6) : 
     
    model = models.resnet18(weights='DEFAULT')
    model.conv1 = nn.Conv2d(4,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.fc = torch.nn.Linear(in_features=512, out_features=1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    models_list.append(model)    
    optimizer_list.append(optimizer)
    loss_list.append(criterion)

n_epochs = 50

train_loss_list = []
valid_loss_list = []

def valid_epoch(test_loader, loss_list, models_list):
    tot_loss, n_samples=[0 for _ in range(6)], [0 for _ in range(6)]
    with torch.no_grad():
        for x_batch, y_batch, _ in test_loader:
            
            for i in range(6) :
                y = feature_extraction(y_batch,i)
                y = torch.tensor(y, dtype=torch.float32)
                models_list[i].eval()
                outputs = models_list[i](x_batch)
                loss = loss_list[i](outputs.squeeze(), y)
                tot_loss[i] += loss.item() * y.size(0)
                n_samples[i] += y.size(0)

    avg_loss = [tot_loss[i] / n_samples[i] if n_samples[i] > 0 else 0.0 for i in range(6)]
    valid_loss_list.append(sum(avg_loss) / len(avg_loss))
    return sum(avg_loss) / len(avg_loss)


final_loss = 0


for epoch in range(n_epochs):

    epoch_loss = [0 for _ in range(6)]
    n_samples = [0 for _ in range(6)]
    
    for x_batch, y_batch, _ in train_freq_data_loader:
        for i in range(6) :
            y = feature_extraction(y_batch,i)
            y = torch.tensor(y, dtype=torch.float32)
            models_list[i].train()
            optimizer_list[i].zero_grad()
            outputs = models_list[i](x_batch)
            loss = loss_list[i](outputs.squeeze(), y)
            loss.backward()
            optimizer_list[i].step()
            epoch_loss[i] += loss.item() * y.size(0)
            n_samples[i] += y.size(0)
        
    
    final_loss = sum(epoch_loss) / sum(n_samples) if sum(n_samples) > 0 else 0.0

    valid_loss = valid_epoch(valid_freq_data_loader, loss_list, models_list)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {(final_loss):.4f}", f"Valid loss; {valid_loss:.4f}")
    train_loss_list.append(final_loss)
    
plt.plot(range(len(train_loss_list)), train_loss_list, label='train')
plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid')
    
#print(f'test mse: {valid_epoch(test_freq_data_loader, criterion, model)}')
#print(f'test mae: {valid_epoch(test_freq_data_loader, nn.L1Loss(), model)}')

plt.legend()
plt.show()