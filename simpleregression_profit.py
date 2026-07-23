import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# FIX 1: Use a raw string (r"...") to prevent unicode escape sequence errors in Windows paths
data = pd.read_csv(r"D:\A4087\Lab-09\linearregressiondataset.csv")

# Extract features (X as 2D DataFrame) and target (y as 1D Series)
x = data.iloc[:, 0:1]
y = data.iloc[:, 1]

print("Features (x):")
print(x.head())
print("\nTarget (y):")
print(y.head())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Creating and fitting the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the coefficients and intercept
print('\nCoefficients:', model.coef_)
print('Intercept:', model.intercept_)

# FIX 2: Pass feature name to single prediction to avoid Scikit-Learn feature name warnings
sample_input = pd.DataFrame([[20.27]], columns=x.columns)
single_pred = model.predict(sample_input)
print('\nProfit for the population of 20.27 lakh is:', single_pred[0])

# Predict on test set
y_pred = model.predict(X_test)
print("\nPredicted values for test set:")
print(y_pred)

# Model evaluation 
print('\n--- Model Evaluation ---')
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))