import torch


class LossRecorder:
    def __init__(self, k):
        self.k = k
        self.top_max_loss = [(0, 0) for _ in range(k)]
        self.top_min_loss = [(1e10, 0) for _ in range(k)]

    def update(self, loss, data):
        self.top_max_loss.append((loss, data))
        self.top_max_loss.sort(key=lambda x: x[0], reverse=True)
        self.top_max_loss = self.top_max_loss[: self.k]
        self.top_min_loss.append((loss, data))
        self.top_min_loss.sort(key=lambda x: x[0])
        self.top_min_loss = self.top_min_loss[: self.k]

    def save(self, path):
        torch.save(
            {
                "top_max_loss": self.top_max_loss,
                "top_min_loss": self.top_min_loss,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.top_max_loss = checkpoint["top_max_loss"]
        self.top_min_loss = checkpoint["top_min_loss"]
