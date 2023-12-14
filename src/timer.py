import time
import torch


class Timer:
    def __init__(self, log_output=False):
        self.start_time = time.time()
        self.log_output = log_output
        self.record_time = 0

    def get_time(self):
        torch.cuda.synchronize()
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        return cost_time

    def log(self, *msg, record=False):
        cost_time = self.get_time()
        if record:
            self.record_time += cost_time
        if self.log_output:
            print(*msg, " cost time: ", cost_time)
