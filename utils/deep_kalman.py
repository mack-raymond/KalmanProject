import torch
import torch.nn as nn
import numpy as np

from kalman_shit import DeepKalmanFilter
import matplotlib.pyplot as plt


class KalmanParameterModel(nn.Module):
    def __init__(self, degree: int = 3, function_type: str = "linear"):
        super().__init__()
        self.degree = degree
        self.transition_matrix = nn.Sequential(
            nn.Linear(3, 32),
            # nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            # nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, degree + 2),
        )
        self.register_buffer("input", torch.zeros(3).reshape(1, -1))
        if function_type == "linear":
            self.input.data[0, 0] = 1
        elif function_type == "exponential":
            self.input.data[0, 1] = 1
        else:
            raise ValueError("Function type not supported")

    def forward(self, dt: float, batch_size: int = 1):
        out = self.transition_matrix(self.input.repeat(batch_size, 1))
        initial_measurement_variance = torch.pow(out[:, -2], 2)
        measurement_variance_smoothing = torch.sigmoid(out[:, -1] + 5)
        transition_matrix = out[:, :-2].reshape(-1, self.degree)
        transition_matrix_ = [transition_matrix]
        for i in range(1, self.degree):
            transition_matrix_.append(torch.roll(transition_matrix, i, 1))
        transition_matrix = torch.stack(transition_matrix_, dim=1)
        # transition_matrix = torch.stack([transition_matrix, torch.roll(transition_matrix, 1, 1), torch.roll(transition_matrix, 2, 1)], dim=1)
        transition_matrix = torch.pow(transition_matrix, 2)
        # torch.eye( self.degree).reshape(1, self.degree, self.degree)
        # transition_matrix *= torch.triu(torch.ones(self.degree, self.degree)).reshape(1, self.degree, self.degree)
        transition_matrix = torch.triu(transition_matrix, diagonal=1)
        transition_matrix += torch.eye(self.degree).reshape(1, self.degree, self.degree)
        return (
            transition_matrix,
            initial_measurement_variance,
            measurement_variance_smoothing,
        )

    # dt = torch.pow(dt, torch.arange(self.degree).reshape(1, 1, -1)).expand(transition_matrix.shape)
    # transition_matrix = (torch.relu(transition_matrix) + 1e-14) @ dt


def generate_exponential_data():
    t = torch.linspace(-1, 20, 300)
    data = (
        torch.exp((t + torch.randn(1) * 0.01) * torch.rand(1) * 0.2)
        + torch.randn_like(t) * torch.randn(1) * 0.7
    )
    return t, data


if __name__ == "__main__":
    model = KalmanParameterModel(degree=3, function_type="exponential")

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(10000):
        x, y = generate_exponential_data()
        x_train, x_pred = x[:200], x[200:]
        y_train, y_pred = y[:200], y[200:]
        x_train_mean, x_train_std = x_train.mean(), x_train.std()
        y_train_mean, y_train_std = y_train.mean(), y_train.std()
        x_train = (x_train - x_train_mean) / x_train_std
        y_train = (y_train - y_train_mean) / y_train_std
        x_pred = (x_pred - x_train_mean) / x_train_std
        y_pred = (y_pred - y_train_mean) / y_train_std
        kalman_filter = DeepKalmanFilter(
            initial_uncertainty=1000000,
            # model=model,
            # transition_matrix=transition_matrix,
            initial_measurement_variance=torch.std(y_train) ** 2,
            # measurement_variance_smoothing=measurement_variance_smoothing,
            degree=3,
        )
        transition_matrix, _, measurement_variance_smoothing = model(
            x_train[1] - x_train[0], 300
        )
        print(transition_matrix[0], measurement_variance_smoothing[0])

        means = []
        variances = []
        measurement_variances = []
        for i in range(0, 200):
            state, variance, measurement_variance = kalman_filter.update(
                y_train[i], transition_matrix[i], measurement_variance_smoothing[i]
            )
            means.append(state[0].reshape(1))
            variances.append(variance[0, 0].reshape(1))
            measurement_variances.append(measurement_variance[0].reshape(1))
        for i in range(200, 300):
            state, variance = kalman_filter.predict_and_store(transition_matrix[i])
            means.append(state[0].reshape(1))
            variances.append(variance[0, 0].reshape(1))
        means = torch.cat(means).reshape(-1, 1)
        variances = torch.cat(variances).reshape(-1, 1)
        target = torch.cat([y_train, y_pred]).reshape(-1, 1)
        measurement_variances = torch.cat(
            [
                torch.cat(measurement_variances).reshape(-1, 1),
                measurement_variances[-1].repeat(100, 1),
            ]
        )
        # variances = torch.clamp(variances, min=1e-14)
        # measurement_variances = torch.clamp(measurement_variances, min=1e-14)
        with torch.no_grad():
            print(torch.mean(means - target))
        # try:
        #     assert torch.all(variances >= 0) and torch.all(measurement_variances >= 0)
        # except AssertionError:
        #     print(variances, measurement_variances)
        #     raise
        loss = -torch.distributions.Normal(
            means, torch.sqrt(variances) + torch.sqrt(measurement_variances) + 1e-10
        ).log_prob(target).mean() + 10 * torch.mean((means - target) ** 2)
        # loss = torch.mean((means - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        if epoch > 989:
            with torch.no_grad():
                x_target = torch.cat([x_train, x_pred]).reshape(-1).numpy()
                target = target.reshape(-1).numpy()
                means = means.reshape(-1).numpy()
                std_dev_var = np.sqrt(variances.reshape(-1).numpy())
                std_dev_meas = np.sqrt(measurement_variances.reshape(-1).numpy())

                plt.plot(x_target, means, label="Mean")
                plt.fill_between(
                    x_target,
                    means - 2 * (std_dev_var + std_dev_meas),
                    means + 2 * (std_dev_var + std_dev_meas),
                    alpha=0.5,
                    label="95% confidence interval",
                )
                plt.scatter(x_target, target, label="Target")
                plt.legend()
                plt.ylim(-2, 15)
                plt.show()
