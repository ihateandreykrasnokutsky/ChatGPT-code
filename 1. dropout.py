import numpy as np

# Dropout function
def dropout_layer(activations, keep_prob=0.5):
    dropout_mask = (np.random.rand(*activations.shape) < keep_prob) / keep_prob
    return activations * dropout_mask

# Example of using dropout
a = np.array([[0.2, 0.5, 0.8], [0.3, 0.7, 0.9]])  # Example activations
dropped_a = dropout_layer(a, keep_prob=0.5)

print("Original Activations:\n", a)
print("After Dropout:\n", dropped_a)
