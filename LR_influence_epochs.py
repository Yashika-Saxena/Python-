import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('epo2.csv')

# Extract 'ANX' and 'EEG epochs' columns
X = df[['ANX']]
y = df['EEG epochs']

# Create a linear regression model
reg = LinearRegression()

# Fit the model
reg.fit(X, y)

# Predict 'EEG epochs'
predictions = reg.predict(X)

# Calculate metrics
mse = mean_squared_error(y, predictions)
r_squared = r2_score(y, predictions)

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")

# Print the trained model parameters
print(f"Slope (Coefficient): {reg.coef_[0]}")
print(f"Intercept: {reg.intercept_}")

# Visualize the linear regression line
plt.scatter(X, y, label='Actual Data')
plt.plot(X, predictions, color='red', label='Linear Regression Line')
plt.title('Linear Regression')
plt.xlabel('ANX')
plt.ylabel('EEG epochs')
plt.legend()
plt.show()
