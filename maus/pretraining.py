import torch
import torch.nn as nn
from torchvision import models
from load_data import train_freq_data_loader, valid_freq_data_loader, test_freq_data_loader
import matplotlib.pyplot as plt

model = models.resnet18(weights='DEFAULT')
model.conv1 = nn.Conv2d(4,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
model.fc = torch.nn.Linear(in_features=512, out_features=1)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
	param.requires_grad = True

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 50

train_loss_list = []
valid_loss_list = []

def valid_epoch(test_loader, loss_func, model):
    
    model.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:

            preds = model(x_batch)

            loss = loss_func(preds.squeeze(), y_batch)
            
            n_samples += y_batch.size(0)
            tot_loss += loss.item() * y_batch.size(0)

    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    valid_loss_list.append(avg_loss)
    return avg_loss

epoch_loss = 0
n_samples = 0

for epoch in range(n_epochs):
    for x_batch, y_batch in train_freq_data_loader:
        model.train()
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * y_batch.size(0)
        n_samples += y_batch.size(0)
        
    valid_loss = valid_epoch(valid_freq_data_loader, criterion, model)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {(epoch_loss/n_samples):.4f}, Valid loss; {valid_loss:.4f}")
    train_loss_list.append(epoch_loss/n_samples)
    
plt.plot(range(len(train_loss_list)), train_loss_list, label='train')
plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid')
    
print(f'test mse: {valid_epoch(test_freq_data_loader, criterion, model)}')
print(f'test mae: {valid_epoch(test_freq_data_loader, nn.L1Loss(), model)}')

plt.legend()
plt.show()