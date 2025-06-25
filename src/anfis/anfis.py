import numpy as np
from .mf import MembershipFunction
from .fls import FuzzyLogicSystem

class ANFIS:
    def __init__(self, n_inputs=2, n_mfs=2, epochs=10, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Initialize membership functions (Gaussian, etc.) for each input
        self.mfs = [[MembershipFunction() for _ in range(n_mfs)] for _ in range(n_inputs)]
        
        # Initialize fuzzy logic system (rule base, output layer)
        self.fls = FuzzyLogicSystem(n_inputs, n_mfs)
    
    def fit(self, X, y):
        # Training loop
        for epoch in range(self.epochs):
            for x_sample, y_sample in zip(X, y):
                # Forward pass: calculate output
                output = self.fls.forward(x_sample, self.mfs)
                # Compute error
                error = y_sample - output
                # Backward pass: update parameters (membership function params and rules)
                self.fls.backward(error, self.mfs, self.learning_rate, x_sample)
            print(f"Epoch {epoch+1}/{self.epochs} completed.")
    
    def predict(self, X):
        preds = []
        for x_sample in X:
            output = self.fls.forward(x_sample, self.mfs)
            preds.append(output)
        return np.array(preds)
