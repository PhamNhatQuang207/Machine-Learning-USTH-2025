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
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X @ theta
    cost = (1/(2*m)) * np.sum((predictions - Y)**2)
    return cost
def gradient_decent(X, Y,alpha, iteration):
    m = len(Y)
    theta = np.zeros((X.shape[1], 1))
    cost_history = []
    for i in range (iteration):
        gradients = (1/m) * (X.T @ (X @ theta - Y))
        theta -= alpha * gradients
        cost = compute_cost(X, Y, theta)
        cost_history.append(cost)
    return theta, cost_history

# Gradient descent parameters
alpha = 0.01
iteration = 1000
theta,cost_history = gradient_decent(X_b, Y, alpha, iteration)

# Print the final theta values and plot the cost vs iterations graph for α = 0.01
print("Final theta values: ")
print(theta)

plt.plot(range(iteration), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations (α = 0.01)")
plt.grid(True)
plt.show()

# New input sample
new_data = np.array([[200, 40, 20]])

# Standardize using previous scaler
new_data_scaled = scaler.transform(new_data)

# Add bias term
new_data_biased = np.c_[np.ones((new_data_scaled.shape[0], 1)), new_data_scaled]

# Predict using theta from normal equation
predicted_sales = new_data_biased @ theta

print("Predicted sales:", predicted_sales[0][0])