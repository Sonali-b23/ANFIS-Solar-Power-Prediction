import numpy as np

class FuzzyLogicSystem:
    def __init__(self, n_inputs, n_mfs):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_inputs
        
        # Initialize consequent parameters randomly (for simplicity)
        self.consequents = np.random.rand(self.n_rules, n_inputs + 1)  # Linear params + bias
    
    def forward(self, x, mfs):
        # Calculate membership degrees for each input and MF
        memberships = []
        for i in range(self.n_inputs):
            memberships.append([mf.compute(x[i]) for mf in mfs[i]])
        
        # Calculate firing strengths of all rules (product of membership degrees)
        firing_strengths = []
        for rule_index in range(self.n_rules):
            strength = 1.0
            for input_idx in range(self.n_inputs):
                # Determine which MF index to use for this input and rule
                mf_index = (rule_index // (self.n_mfs ** input_idx)) % self.n_mfs
                strength *= memberships[input_idx][mf_index]
            firing_strengths.append(strength)
        
        firing_strengths = np.array(firing_strengths)
        
        # Normalize firing strengths
        norm_strengths = firing_strengths / np.sum(firing_strengths)
        
        # Compute rule outputs (linear functions of inputs + bias)
        rule_outputs = []
        for i in range(self.n_rules):
            params = self.consequents[i]
            rule_output = np.dot(params[:-1], x) + params[-1]  # Linear combination + bias
            rule_outputs.append(rule_output)
        rule_outputs = np.array(rule_outputs)
        
        # Final output is weighted sum of rule outputs
        output = np.dot(norm_strengths, rule_outputs)
        return output

    def backward(self, error, mfs, lr, x):
        # For each rule, update the consequent params by gradient descent on error
        # We'll approximate the gradient by: gradient = -error * normalized firing strength * inputs
        
        # Calculate membership degrees for each input and MF
        memberships = []
        for i in range(self.n_inputs):
            memberships.append([mf.compute(x[i]) for mf in mfs[i]])
        
        # Compute firing strengths of rules (as in forward pass)
        firing_strengths = []
        for rule_index in range(self.n_rules):
            strength = 1.0
            for input_idx in range(self.n_inputs):
                mf_index = (rule_index // (self.n_mfs ** input_idx)) % self.n_mfs
                strength *= memberships[input_idx][mf_index]
            firing_strengths.append(strength)
        
        firing_strengths = np.array(firing_strengths)
        norm_strengths = firing_strengths / np.sum(firing_strengths)
        
        # Update consequent params for each rule
        for i in range(self.n_rules):
            # Gradient update for linear coefficients and bias
            # For linear coeffs
            self.consequents[i][:-1] += lr * error * norm_strengths[i] * x  # x is current sample input
            # For bias
            self.consequents[i][-1] += lr * error * norm_strengths[i]

