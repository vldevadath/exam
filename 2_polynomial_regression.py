import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/home/KasiR/Downloads/auto-mpg.csv").dropna()

X = df[['displacement']]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

linear_mse = mean_squared_error(y_test, y_pred_lin)
linear_r2 = r2_score(y_test, y_pred_lin)
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_r2 = r2_score(y_test, y_pred_poly)

print("Linear Regression")
print("MSE:", round(linear_mse, 2))
print("R2 Score:", round(linear_r2, 2))

print("\nPolynomial Regression (Degree 2)")
print("MSE:", round(poly_mse, 2))
print("R2 Score:", round(poly_r2, 2))

plt.scatter(X, y, label="actual data", alpha=0.2)
X_range = np.linspace(X['displacement'].min(), X['displacement'].max(), 100).reshape(-1, 1)
X_range_df = pd.DataFrame(X_range, columns=['displacement'])
plt.plot(X_range, lin_reg.predict(X_range_df), color="blue", label="linear reg")
plt.plot(X_range, poly_reg.predict(poly_features.transform(X_range_df)), color="red", label="Poly Reg")
plt.legend()
plt.grid(True)
plt.show()
