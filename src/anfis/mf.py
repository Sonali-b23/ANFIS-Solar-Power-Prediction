import numpy as np

class MembershipFunction:
    def __init__(self, mean=0, sigma=1):
        self.mean = mean
        self.sigma = sigma
    
    def compute(self, x):
        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)
    
    def update(self, delta_mean, delta_sigma, lr):
        self.mean += lr * delta_mean
        self.sigma += lr * delta_sigma
