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


resnet = models.resnet18(weights='DEFAULT')
resnet.conv1 = nn.Conv2d(4,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
resnet.fc = torch.nn.Linear(in_features=512, out_features=1)

for param in resnet.parameters():
    param.requires_grad = False

for param in resnet.fc.parameters():
	param.requires_grad = True

class FCNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # separates all features in separate branches 
        self.branches = nn.ModuleList([resnet for _ in range(6)])


    def forward(self, x_batch, weight_batch):

        branch_outputs = []

        for branch in self.branches:
            branch_outputs.append(branch(x_batch))  

        branch_outputs = torch.stack(branch_outputs).float()
        branch_outputs = branch_outputs.permute(1, 0, 2)
        weights = weight_batch.float().unsqueeze(-1)
        weighted = (branch_outputs * weights).sum(dim=1) / weights.sum(dim=1)

        return weighted   

model = FCNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50

train_loss_list = []
valid_loss_list = []


def valid_epoch(test_loader, loss_func, model):
    model.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for x_batch, y_batch, weight_batch in test_loader:

            preds = model(x_batch, weight_batch)

            loss = loss_func(preds.squeeze(), y_batch)
            
            n_samples += y_batch.size(0)
            tot_loss += loss.item() * y_batch.size(0)

    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    valid_loss_list.append(avg_loss)
    return avg_loss

for epoch in range(n_epochs):
    epoch_loss = 0
    n_samples = 0
    for x_batch, y_batch, weight_batch in train_freq_data_loader:
        model.train()
        optimizer.zero_grad()
        outputs = model(x_batch, weight_batch)
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