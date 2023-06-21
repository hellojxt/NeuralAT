import commentjson as json
import tinycudann as tcnn
import torch
from tqdm import tqdm

with open("config/config.json") as f:
    config = json.load(f)


class Signal(torch.nn.Module):
    def __init__(self, frequency, amplitude):
        super().__init__()
        self.frequency = frequency
        self.amplitude = amplitude

    def forward(self, x):
        with torch.no_grad():
            return self.amplitude * torch.sin(self.frequency * 2 * torch.pi * x)


encoding = tcnn.Encoding(1, config["encoding"])
network = tcnn.Network(encoding.n_output_dims, 1, config["network"])
model = torch.nn.Sequential(encoding, network)

sig = Signal(200, 1).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 64000

for i in tqdm(range(2000)):
    optimizer.zero_grad()
    x = torch.rand([batch_size, 1], device="cuda")
    y = sig(x).float()
    y_pred = model(x).float()
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(loss.item())

import matplotlib.pyplot as plt

x = torch.linspace(0, 0.2, 1000, device="cuda").reshape(-1, 1)
y = sig(x).float()
y_pred = model(x).float()
print(x.shape, y.shape, y_pred.shape)
plt.plot(x.cpu().numpy(), y.cpu().numpy(), label="True")
plt.plot(x.cpu().numpy(), y_pred.cpu().detach().numpy(), label="Predicted")

plt.legend()
plt.show()
