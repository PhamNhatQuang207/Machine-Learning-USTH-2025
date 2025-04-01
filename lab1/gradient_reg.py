import pandas as pd;
import numpy as np;
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Advertising.csv",index_col=0)
X = df[['TV','radio','newspaper']].values
Y = df[['sales']].values

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add a column of ones to the scaled features to include bias term in the linear regression model
X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# Define cost function and gradient descent algorithm
def compute_cost(X, Y, theta, reg):
    m = len(Y)
    predictions = X @ theta
    cost = (1/(2*m)) * np.sum((predictions - Y)**2 + reg * np.sum(theta[1:]**2))
    return cost
def gradient_decent(X, Y,alpha, iteration, reg):
    m = len(Y)
    theta = np.zeros((X.shape[1], 1))
    cost_history = []
    for i in range (iteration):
        cost = compute_cost(X, Y, theta ,reg)
        gradients = (1 / m) * (X.T @ (X @ theta - Y))
        gradients[1:] += (reg / m) * theta[1:]  
        theta -= alpha * gradients
        cost_history.append(cost)
    return theta, cost_history

# Gradient descent parameters
alpha = 0.01
iteration = 1000
reg = 0.01 # Regularization parameter
theta,cost_history = gradient_decent(X_b, Y, alpha, iteration,reg)

# Print the final theta values and plot the cost vs iterations graph for α = 0.01
print("Final theta values: ")
print(theta)
