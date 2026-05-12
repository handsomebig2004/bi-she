import torch
from torchview import draw_graph
from rnn import MultiSignalRNN

# 你的模型
model = MultiSignalRNN(num_signals=3, hidden_size=128, out_dim=6)

# 构造一个假的输入：x_list = [ecg, gsr, inf_ppg]，每个 [B, L]
B, L = 2, 7680
dummy = [torch.randn(B, L), torch.randn(B, L), torch.randn(B, L)]

# 画图
g = draw_graph(
    model,
    input_data=(dummy,),          # 注意：forward 接收的是一个参数 x_list，所以要包一层 tuple
    expand_nested=True,
    graph_name="MultiSignalRNN",
    device="cpu"
)
