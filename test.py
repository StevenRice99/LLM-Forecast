import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Given list of numbers
numbers = [1, 3, 1]

# Convert the list to a numpy array
y = np.array(numbers)

# Define the ARIMA model (order=(p,d,q))
# Here p=1, d=1, q=1 is a simple starting point
model = ARIMA(y, order=(1, 1, 1))

# Fit the model
model_fit = model.fit()

# Predict the next value
next_value = model_fit.forecast(steps=1)
print(f"The predicted next value using ARIMA is: {next_value[0]}")
