import torch

b = torch.load("b.pt").reshape(-1).detach().cpu().numpy()
neumann = torch.load("neumann.pt").reshape(-1).detach().cpu().numpy()
predict = torch.load("predict.pt").reshape(-1).detach().cpu().numpy()
import matplotlib.pyplot as plt

fig = plt.figure()
# plot histogram

ax = fig.add_subplot(111)
ax.plot(b, label="b")
ax.plot(predict, label="predict")
plt.show()
