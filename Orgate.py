import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [1]])

np.random.seed(42)  
weights = np.random.uniform(-1, 1, (2, 1))
bias = np.random.uniform(-1, 1, (1,))

learning_rate = 0.1

for epoch in range(10000):
    linear_output = np.dot(inputs, weights) + bias
    predictions = sigmoid(linear_output)

    error = outputs - predictions

    d_pred = error * sigmoid_derivative(predictions)
    weights += np.dot(inputs.T, d_pred) * learning_rate
    bias += np.sum(d_pred) * learning_rate

print("\nFinal Weights:", weights)
print("Final Bias:", bias)

print("\nPredictions:")
for i in range(len(inputs)):
    test_output = sigmoid(np.dot(inputs[i], weights) + bias)
    print(f"Input: {inputs[i]} => Predicted Output: {round(test_output[0])}")
