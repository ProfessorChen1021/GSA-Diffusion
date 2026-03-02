import time


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def reset(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0
