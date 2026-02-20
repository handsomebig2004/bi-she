from fcn import FCNModel, train, test_model
import matplotlib.pyplot as plt
import torch
import rnn
import load_data


fcn_net=FCNModel(num_signals=3, kernel_size=7)
rnn_net = rnn.MultiSignalRNN(num_signals=3)

loss_func = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
optim_adam = torch.optim.Adam(params= fcn_net.parameters())

train_loss_list, valid_loss_list = train(fcn_net, load_data.train_data_loader, load_data.valid_data_loader, loss_func, optim_adam, n_epochs=15)
plt.plot(range(len(train_loss_list)), train_loss_list, label='train')

print(f'test loss (mse): {test_model(fcn_net, load_data.test_data_loader, loss_func=loss_func)}')
print(f'test loss (mae): {test_model(fcn_net, load_data.test_data_loader, loss_func=mae_loss)}')

plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid')
plt.legend()
plt.show()

