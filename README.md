# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the dataset by importing the required libraries, reading the CSV file, applying one-hot encoding to the categorical State column, and standardizing the input features and target values.

2.Prepare the data for training by separating the independent features (X) and target variable (y), then adding a bias (intercept) column to the feature matrix.

3.Train the Linear Regression model using Batch Gradient Descent by initializing the parameters, repeatedly updating them to minimize the Mean Squared Error (MSE) cost function, and plotting the cost convergence graph.

4.Predict the profit for a new startup by preprocessing the new input in the same way as the training data, applying the learned parameters to make a prediction, and converting the predicted value back to the original scale.


## Program:
```python
/*
Program to implement Linear Regression using Gradient Descent on 50 Startups Dataset.
Developed by: K SriSaran Karthik
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("50_Startups.csv")

print("First 5 rows of the dataset:")
display(data.head())

data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)

print("Encoded dataset:")
display(data_encoded.head())

X_raw = data_encoded.drop('Profit', axis=1).values
y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)

m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))

print("Shape of X after adding bias term:", X.shape)

def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost

def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history

alpha = 0.01

theta, J_hist = gradient_descent(
    X, y,
    learning_rate=alpha,
    num_iters=5000,
    tol=1e-9,
    verbose=True
)

print("\nLearned Parameters (Theta):")
print(theta.flatten())

print("Developed by: K SriSaran Karthik")

plt.figure(figsize=(7, 4))
plt.plot(J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()

new_sample = pd.DataFrame([{
    'R&D Spend': 165349.2,
    'Administration': 136897.8,
    'Marketing Spend': 471784.1,
    'State': 'New York'
}])

# Apply one-hot encoding
new_encoded = pd.get_dummies(
    new_sample,
    columns=['State'],
    drop_first=True
)

# Align columns with the training data
new_encoded = new_encoded.reindex(
    columns=data_encoded.drop('Profit', axis=1).columns,
    fill_value=0
)

# Scale the new data
new_scaled = X_scaler.transform(new_encoded)

# Add bias term
new_design = np.hstack((
    np.ones((new_scaled.shape[0], 1)),
    new_scaled
))

# Predict scaled profit
scaled_pred = new_design.dot(theta)

# Convert prediction to original units
pred_original = y_scaler.inverse_transform(scaled_pred)

print(
    f"\nPredicted Profit for the new startup: "
    f"₹{pred_original[0][0]:,.2f}"
)
```

## Output:
<img width="646" height="261" alt="Screenshot 2026-10-29 142808" src="https://github.com/user-attachments/assets/264bdaa0-6031-4883-bc4c-f3a1f7e69228" />
<img width="856" height="263" alt="Screenshot 2026-10-29 142815" src="https://github.com/user-attachments/assets/0575900a-79f9-4adb-8514-6d0c7c6f11a0" />
<img width="652" height="347" alt="Screenshot 2026-10-29 142826" src="https://github.com/user-attachments/assets/806c6842-c15a-4376-a850-3b1da628cf0d" />
<img width="926" height="508" alt="Screenshot 2026-10-29 142836" src="https://github.com/user-attachments/assets/740bd27c-c3c5-4051-9b57-175544fd3381" />
<img width="497" height="50" alt="Screenshot 2026-10-29 142847" src="https://github.com/user-attachments/assets/e4d97347-b7bd-4986-ba6d-fdaa5d84b28d" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
