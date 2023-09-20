import numpy as np
from kalman_filter import KalmanFilter

# Initialize the Kalman filter.
x_hat = np.array([1, 2])
P = np.array([[1, 0], [0, 1]])
F = np.array([[1, 0], [0, 1]])
H = np.array([[1, 0]])
R = np.array([[0.1]])
Q = np.array([[0.01, 0], [0, 0.01]])

kalman_filter = KalmanFilter(x_hat, P, F, H, R, Q)

# Predict the next state vector and covariance matrix.
kalman_filter.predict()

# Update the state vector and covariance matrix based on a measurement.
z = np.array([3])
kalman_filter.update(z)

# Get the updated state vector and covariance matrix.
updated_x_hat = kalman_filter.x_hat
updated_P = kalman_filter.P

print(updated_x_hat)
print(updated_P)