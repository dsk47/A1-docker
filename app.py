import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)   # independent variable
y = np.array([2, 4, 5, 4, 5])                  # dependent variable

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)
print("Predicted values:", y_pred)

print("Slope (m):", model.coef_)
print("Intercept (b):", model.intercept_)

plt.scatter(X, y, color='blue')      # original data
plt.plot(X, y_pred, color='red')     # regression line
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()
