import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/home/KasiR/Downloads/California Housing Dataset — Scikit-learn CSV Expo/california_housing.csv")

X = df['AveRooms'].values.reshape(-1, 1)
y = df['MedHouseVal'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

w = 0
b = 0
LR = 0.001
n = len(X_train)

for i in range(1000):
    y_pred = w * X_train.flatten() + b
    dw = (-2/n) * np.sum(X_train.flatten() * (y_train - y_pred))
    db = (-2/n) * np.sum(y_train - y_pred)
    w = w - LR * dw
    b = b - LR * db

y_pred_gd = w * X_test.flatten() + b

print("\nGradient Descent")
print("Slope =", w)
print("Intercept =", b)
print("MSE =", mean_squared_error(y_test, y_pred_gd))
print("R2 Score =", r2_score(y_test, y_pred_gd))

X_train_new = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_new = np.c_[np.ones((len(X_test), 1)), X_test]

theta = np.linalg.inv(X_train_new.T @ X_train_new) @ X_train_new.T @ y_train
y_pred_ne = X_test_new @ theta

print("\nNormal Equation")
print("Intercept =", theta[0])
print("Slope =", theta[1])
print("MSE =", mean_squared_error(y_test, y_pred_ne))
print("R2 Score =", r2_score(y_test, y_pred_ne))

plt.scatter(X, y, label="Actual data", alpha=0.4)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_new = np.c_[np.ones((len(X_range), 1)), X_range]
y_range = X_range_new @ theta
plt.plot(X_range, y_range, color="blue", label="Linear regression")
plt.xlabel("Average rooms")
plt.ylabel("House Value")
plt.legend()
plt.grid(True)
plt.show()
