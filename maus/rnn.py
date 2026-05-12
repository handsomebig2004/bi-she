import torch
import torch.nn as nn

def _prepare(y):
    if isinstance(y, (list, tuple)):
        y = y[0]
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    if y.dim() == 0:
        y = y.unsqueeze(0)
    return y.float()


class MultiSignalRNN(nn.Module):
    """
    Multi-signal LSTM for regression.
    Input: x_list = [ecg, gsr, inf_ppg], each tensor shape [B, L]
    Output: y_hat shape [B, 6]
    """

    def __init__(self, num_signals=3, hidden_size=128, num_layers=1, dropout=0.0, out_dim=6):
        super().__init__()
        self.num_signals = num_signals

        self.rnn = nn.LSTM(
            input_size=num_signals,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_list):
        # x_list: list of length num_signals, each [B, L]
        x_list = [_to_float_tensor(x) for x in x_list]

        # Build [B, L, num_signals]
        feats = []
        for x in x_list:
            # If someone accidentally gives [B,1,L], squeeze it
            if x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)
            feats.append(x.unsqueeze(-1))  # [B, L, 1]
        x = torch.cat(feats, dim=-1)       # [B, L, num_signals]

        out, _ = self.rnn(x)              # [B, L, hidden]
        pooled = out.mean(dim=1)          # [B, hidden]  (global average over time)
        y_hat = self.fc(pooled)           # [B, 6]
        return y_hat



def epoch_train_rnn(net, train_loader_l, y_train_loader, loss_func, optim):
    net.train()
    tot_loss, n_samples = 0.0, 0

    for *x_batches, y_batch in zip(*train_loader_l, y_train_loader):
        optim.zero_grad()

        y = _prepare(y_batch)

        preds = net(x_batches).squeeze()   # [batch] or [batch, 1] -> [batch]

        loss = loss_func(preds, y.float())
        loss.backward()
        optim.step()

        batch_size = y.size(0)
        n_samples += batch_size
        tot_loss += loss.item() * batch_size

    return tot_loss / n_samples


def epoch_valid_rnn(net, valid_loader_l, y_valid_loader, loss_func):
    net.eval()
    tot_loss, n_samples = 0.0, 0


    with torch.no_grad():
        for *x_batches, y_batch in zip(*valid_loader_l, y_valid_loader):
            y = _prepare(y_batch)
            preds = net(x_batches).squeeze()
            loss = loss_func(preds, y.float())

            batch_size = y.size(0)
            n_samples += batch_size
            tot_loss += loss.item() * batch_size

    return tot_loss / n_samples if n_samples > 0 else 0.0


def train_rnn(net, train_loader_l, valid_loader_l, y_train_loader, y_valid_loader,
              loss_func, optim, n_epochs):
    train_loss_list, valid_loss_list = [], []

    for epoch in range(n_epochs):
        # -------- train --------
        net.train()
        tot_loss, n = 0.0, 0

        for x_batch, y_batch in train_loader:
            optim.zero_grad()

            # x_batch is a tuple: (ecg, gsr, inf_ppg)
            x_list = [xb.to(device).float() for xb in x_batch]
            y = y_batch.to(device).float()  # [B,6]

            preds = net(x_list)             # [B,6]
            loss = loss_func(preds, y)

            loss.backward()
            optim.step()

            bsz = y.size(0)
            tot_loss += loss.item() * bsz
            n += bsz

        train_loss = tot_loss / max(n, 1)
        train_loss_list.append(train_loss)

        # -------- valid --------
        net.eval()
        tot_loss, n = 0.0, 0

        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_list = [xb.to(device).float() for xb in x_batch]
                y = y_batch.to(device).float()

                preds = net(x_list)
                loss = loss_func(preds, y)

                # 打印一次样例：和你 FCN 的 epoch_valid 里打印 preds[:5], y[:5] 对齐
                if print_samples and epoch == 0 and tot_loss == 0.0:
                    print(preds[:5].cpu())
                    print(y[:5].cpu())

                bsz = y.size(0)
                tot_loss += loss.item() * bsz
                n += bsz

        valid_loss = tot_loss / max(n, 1)
        valid_loss_list.append(valid_loss)

        print(f"[RNN] Epoch {epoch}:  train loss {train_loss}, valid loss {valid_loss}")

    return train_loss_list, valid_loss_list
