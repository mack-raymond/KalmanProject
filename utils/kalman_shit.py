import numpy as np
import torch.nn as nn
import torch
from scipy.special import factorial


class KalmanFilter:
    def __init__(self, initial: float, initial_uncertainty: float,
                 process_variance: float = None, measurement_variance: float = None):
        self.process_variance = process_variance
        self.x = np.array([0, 0, 0]).reshape(-1, 1)
        self.P = np.eye(3) * initial_uncertainty
        self.H = np.array([1, 0, 0]).reshape(1, -1)
        if measurement_variance is not None:
            self.R = np.array([[measurement_variance]])

    def F(self, dt: float):
        F = np.array([[1, dt, dt ** 2 / 2],
                      [0, 1, dt],
                      [0, 0, 1]])
        return F

    def Q(self, dt: float, F: np.ndarray = None):
        Q = np.array([[dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
                      [dt ** 3 / 2, dt ** 2, dt],
                      [dt ** 2 / 2, dt, 1]])
        if F is None:
            F = self.F(dt)
        Q = F @ Q @ F.T
        return Q * self.process_variance

    def predict(self, dt: float):
        F = self.F(dt)
        Q = self.Q(dt, F)

        next_x = F @ self.x
        next_P = F @ self.P @ F.T + Q

        return next_x, next_P, Q

    def predict_and_store(self, dt: float):
        self.x, self.P, Q = self.predict(dt)
        return self.x, self.P, Q

    def update(self, measurement: float, dt: float, measurement_variance: float = None, process_variance: float = None):
        if measurement_variance is not None:
            self.R = np.array([[measurement_variance]])
        if process_variance is not None:
            self.process_variance = process_variance
        if self.R is None:
            raise ValueError('Measurement variance is not set')
        if self.process_variance is None:
            raise ValueError('Process variance is not set')

        x, P, Q = self.predict(dt)
        K = P @ self.H.T @ np.linalg.inv(self.H @ P @ self.H.T + self.R)
        self.x = x + K @ (measurement - self.H @ x)
        A_ = np.eye(3) - K @ self.H
        self.P = A_ @ P @ A_.T + K @ self.R @ K.T

        return x, P


class DeepKalmanFilter(nn.Module):
    def __init__(self,
                 initial_uncertainty: float,
                 initial_measurement_variance: [float, torch.Tensor],
                 degree: int = 3,
                 ):
        super().__init__()

        self.degree = degree
        self.state = torch.zeros(degree).reshape(-1, 1)
        self.P = torch.eye(degree) * initial_uncertainty
        self.H = torch.zeros(degree).reshape(1, -1)
        self.H.data[0, 0] = 1

        if isinstance(initial_measurement_variance, float):
            self.register_buffer('R', torch.tensor([[initial_measurement_variance]]))
        else:
            self.R = initial_measurement_variance.reshape(1, 1)

    def F(self, dt: float):
        F = [torch.pow(dt, torch.arange(self.degree)) / factorial(torch.arange(self.degree))]
        for i in range(1, self.degree):
            F.append(torch.roll(F[-1], 1))
        F = torch.stack(F, dim=0)
        F = torch.triu(F)

        return F

    def predict(self, transition_matrix: torch.Tensor):
        transition_matrix = self.F(1.)

        next_x = transition_matrix @ self.state.detach()
        next_P = transition_matrix @ self.P.detach() @ transition_matrix.T

        return next_x, next_P

    def predict_and_store(self, transition_matrix: torch.Tensor):
        transition_matrix = self.F(1.)
        self.state, self.P = self.predict(transition_matrix)
        return self.state, self.P

    def update(self, measurement: float, transition_matrix: torch.Tensor, meas_var_smoothing: torch.Tensor):
        transition_matrix = self.F(1.)
        self.R = self.R.detach()
        self.H = self.H.detach()
        self.state = self.state.detach()
        self.P = self.P.detach()

        x, P = self.predict(transition_matrix)
        self.R = (measurement - self.H @ x) ** 2 * (1 - meas_var_smoothing) + self.R * meas_var_smoothing
        K = P @ self.H.T @ torch.inverse(self.H @ P @ self.H.T + self.R)
        print('--------------')
        print(P @ self.H.T)
        print(torch.inverse(self.H @ P @ self.H.T + self.R))
        print(K)
        print('--------------')
        try:
            assert torch.all(K >= 0)
        except AssertionError:
            print(K)
            print(self.R)
            raise
        self.state.data = x + K @ (measurement - self.H @ x)
        A_ = torch.eye(self.degree) - K @ self.H
        self.P = A_ @ P @ A_.T + K @ self.R @ K.T

        return x, P, self.R


