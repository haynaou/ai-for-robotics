import numpy as np

class KalmanFilter:
  """A Kalman filter class.

  Attributes:
    x_hat: The estimated state vector.
    P: The estimated state covariance matrix.
    F: The state transition matrix.
    H: The measurement matrix.
    R: The measurement noise covariance matrix.
    Q: The process noise covariance matrix.
    x_hat_pred: The predicted state vector.
    P_pred: The predicted state covariance matrix.
  """

  def __init__(self, x_hat, P, F, H, R, Q):
    """Initializes the Kalman filter.

    Args:
      x_hat: The initial estimated state vector.
      P: The initial estimated state covariance matrix.
      F: The state transition matrix.
      H: The measurement matrix.
      R: The measurement noise covariance matrix.
      Q: The process noise covariance matrix.
    """

    self.x_hat = x_hat
    self.P = P
    self.F = F
    self.H = H
    self.R = R
    self.Q = Q

  def predict(self):
    """Predicts the next state vector and covariance matrix.

    Returns:
      None.
    """

    # Predict the next state vector.
    self.x_hat_pred = self.F * self.x_hat

    # Predict the next state covariance matrix.
    self.P_pred = self.F * self.P * self.F.T + self.Q

  def update(self, z):
    """Updates the state vector and covariance matrix based on a measurement.

    Args:
      z: The measurement.

    Returns:
      None.
    """
    # Compute innovation covariance matrix.
    s = self.H * self.P_pred * self.H.T + self.R

    # Compute the Kalman gain.
    K = self.P_pred * self.H.T * np.linalg.inv(s)

    # Compute the innovation.
    y = z - self.H * self.x_hat_pred

    # Update the state vector.
    self.x_hat = self.x_hat_pred + K * y

    # Update the state covariance matrix.
    self.P = (np.eye(self.x_hat.shape[0]) - K * self.H) * self.P_pred
    
    