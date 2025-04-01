import pandas as pd;
import numpy as np;
from sklearn.preprocessing import StandardScaler

# Load dataset and preprocess data
df = pd.read_csv("Advertising.csv",index_col=0)
X = df[['TV','radio','newspaper']].values
Y = df[['sales']].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add bias term to X (for linear regression)
X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# Normal Equation implementation
def normal_equation(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

theta = normal_equation(X_b, Y)

# Print the final theta values
print("Final theta values: ")
print(theta)

# New input sample
new_data = np.array([[150, 25, 20]])

# Standardize using previous scaler
new_data_scaled = scaler.transform(new_data)

# Add bias term
new_data_biased = np.c_[np.ones((new_data_scaled.shape[0], 1)), new_data_scaled]

# Predict using theta from normal equation
predicted_sales = new_data_biased @ theta

print("Predicted sales:", predicted_sales[0][0])
