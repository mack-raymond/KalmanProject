import torch.nn as nn
import torch
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from data_generators import (
    generate_exponential_data,
    generate_sin_data,
    generate_cubic_data,
)


class KalmanFilter(nn.Module):
    def __init__(
        self, initial_P: torch.Tensor, degree: int = 3, batch_size: int = 1,
    ):
        super().__init__()

        self.degree = degree
        self.batch_size = batch_size
        self.state = (
            torch.zeros(degree).float().reshape(1, -1, 1).repeat(batch_size, 1, 1)
        )
        self.P = (
            torch.eye(degree)
            .float()
            .reshape(1, degree, degree)
            .repeat(batch_size, 1, 1)
            * 10000000
        )
        # self.P = initial_P.float().reshape(1, degree, degree).repeat(batch_size, 1, 1)
        # print(self.P)
        self.H = torch.zeros(degree).float().reshape(1, 1, -1).repeat(batch_size, 1, 1)
        self.H.data[:, 0, 0] = 1

    def Q(self, dt: torch.Tensor):
        F = self.F(dt)
        # print(torch.stack([dt ** 2 / 2, dt ** 3 / 6, dt ** 4 / 24], dim=1).shape)
        Q = torch.stack(
            [
                torch.stack([dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2], 1),
                torch.stack([dt ** 3 / 2, dt ** 2, dt], 1),
                torch.stack([dt ** 2 / 2, dt, torch.ones(len(dt))], 1),
            ],
            2,
        )
        # print(F.shape, Q.shape, torch.transpose(F, 1, 2).shape)
        Q = F @ Q @ torch.transpose(F, 1, 2)
        return Q

    #
    def F(self, dt: torch.Tensor):
        if dt.ndim == 1:
            dt = dt.reshape(-1, 1)
        F = [
            torch.pow(dt, torch.arange(self.degree).reshape(1, -1))
            / factorial(torch.arange(self.degree).reshape(1, -1))
        ]
        for i in range(1, self.degree):
            F.append(torch.roll(F[-1], 1, dims=1))
        F = torch.stack(F, dim=1)
        F = torch.triu(F).float()
        return F

    def predict(self, dt: float, Q: torch.Tensor):
        assert Q.ndim == 3
        F = self.F(dt)

        next_x = F @ self.state
        next_P = F @ self.P @ torch.transpose(F, 1, 2) + Q

        return next_x, next_P

    def predict_and_store(self, dt: float, Q: torch.Tensor):
        self.x, self.P = self.predict(dt, Q)
        return self.x, self.P

    def update(
        self,
        measurement: torch.Tensor,
        dt: torch.Tensor,
        R: torch.Tensor,
        Q: torch.Tensor,
    ):
        Q = self.Q(dt) * 0.001
        R = torch.tensor([[[1000.0]]]).float().repeat(self.batch_size, 1, 1)
        # R = R.detach()
        x, P = self.predict(dt, Q)
        H_T = torch.transpose(self.H, 1, 2)
        K = P @ H_T @ torch.inverse(self.H @ P @ H_T + R)
        # print(K)
        measurement = measurement.reshape(-1, 1, 1)
        # print(x.shape, K.shape, measurement.shape, self.H.shape)
        self.state = x + K @ (measurement - self.H @ x)
        A_ = torch.eye(self.degree).repeat(self.batch_size, 1, 1) - K @ self.H

        self.P = A_ @ P @ torch.transpose(A_, 1, 2) + K @ R @ torch.transpose(K, 1, 2)
        # print(self.P, self.state)

        return x, P


class DeepKalmanParameters(nn.Module):
    def __init__(self, hidden_size: int, degree: int):
        super().__init__()
        self.degree = degree
        self.encoder = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, degree ** 2 + 1),
        )
        self.register_parameter(
            "initial_P", nn.Parameter(torch.exp(torch.rand(1, degree, degree)))
        )

    def forward(self, x) -> "(torch.Tensor, torch.Tensor)":
        batch_size, seq_len, _ = x.shape
        x = self.encoder(x)
        x, _ = self.gru(x)
        x = self.decoder(x)
        Q = torch.exp(x[:, :, : self.degree ** 2]).reshape(
            batch_size, seq_len, self.degree, self.degree
        )
        R = torch.exp(x[:, :, -1]).reshape(batch_size, seq_len, 1, 1)

        return self.initial_P, Q, R


def normalize_input(
    x, y, split
) -> "(np.ndarray, np.ndarray, [np.ndarray, np.ndarray], [np.ndarray, np.ndarray])":
    if x.ndim == 3:
        split = int(split * x.shape[1])
        x_stats = x_mean, x_std = (
            np.mean(x[:, :split], 1, keepdims=True),
            np.std(x[:, :split], 1, keepdims=True),
        )
        y_stats = y_mean, y_std = x_mean[:, :, 0, None], x_std[:, :, 0, None]

    else:
        split = int(split * x.shape[0])
        x_stats = x_mean, x_std = x[:split].mean(0), x[:split].std(0)
        y_stats = y_mean, y_std = x_mean[0], x_std[0]
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    return x, y, x_stats, y_stats


def scale_outputs(x, y, x_stats, y_stats):
    x_mean, x_std = x_stats
    y_mean, y_std = y_stats

    x = x * x_std + x_mean
    y = y * y_std + y_mean
    return x, y


if __name__ == "__main__":
    total_epochs = 1000
    generated_points = 10000
    max_sampled_points = 1000
    max_removed_points = 10
    stats_split = 1
    degree = 3
    batch_size = 1

    model = DeepKalmanParameters(32, degree)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.GaussianNLLLoss(reduction="mean", eps=1e-6)
    # model.cuda()

    model.train()
    # fig, ax = plt.subplots(5, 5, figsize=(20, 20))

    ax_counter = 0
    for epoch in range(total_epochs):
        x_trains = []
        y_trains = []

        sampled_points = np.random.randint(10, max_sampled_points, 1)[0]
        for _ in range(batch_size):
            generate = np.random.choice(
                [generate_cubic_data, generate_exponential_data, generate_sin_data]
            )
            time, measurement = generate(
                generated_points, np.random.randint(0, max_removed_points, 1)[0]
            )
            indices = np.random.choice(generated_points, sampled_points, replace=False)
            indices = np.sort(indices)
            time, measurement = time[indices], measurement[indices]

            dt = time[1:] - time[:-1]

            initial_measurement = measurement[:-1]
            measurement = measurement[1:]
            x_train = np.stack([initial_measurement, dt], axis=1)
            y_train = measurement.reshape(-1, 1)
            x_trains.append(x_train)
            y_trains.append(y_train)
        x_trains = np.stack(x_trains, axis=0)
        y_trains = np.stack(y_trains, axis=0)

        x_positions = x_trains[:, :, 0, None]
        dts = x_trains[:, :, 1, None]
        x_positions, y_train, x_stats, y_stats = normalize_input(
            x_positions, y_trains, stats_split
        )
        x_train = np.concatenate([x_positions, dts], axis=2)
        x_train, y_train = (
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).float(),
        )

        initial_P, Q, R = model(x_train)
        kalman_filter = KalmanFilter(initial_P, degree, batch_size=batch_size)

        pred_means = []
        pred_vars = []
        for i in range(len(y_train[0])):
            state, P = kalman_filter.update(
                y_train[:, i], x_train[:, i, 1], R[:, i], Q[:, i]
            )
            pred_means.append(state[:, 0, 0])
            pred_vars.append(P[:, 0, 0])
        pred_means = torch.stack(pred_means, 1)
        pred_vars = torch.stack(pred_vars, 1)
        R = R[:, :, 0, 0]
        pred_stds = torch.sqrt(pred_vars)

        y_train = y_train[:, :, 0]
        loss = loss_fn(pred_means, y_train, pred_stds)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # print(f"Epoch {epoch} loss: {loss.item()}")
        if epoch % 100 == 0:
            with torch.no_grad():
                y_train = y_train.cpu().numpy()[0]
                pred_means = pred_means.cpu().numpy()[0]
                pred_stds = pred_stds.cpu().numpy()[0]
                plt.plot(np.arange(len(pred_means)), y_train, label="pred")
                plt.fill_between(
                    np.arange(len(pred_means)),
                    pred_means - 2 * pred_stds,
                    pred_means + 2 * pred_stds,
                    alpha=0.5,
                )
                plt.scatter(np.arange(len(y_train)), y_train, label="True")
                plt.legend()
                plt.ylim(-2, 2)
                plt.show()

    #     if (epoch + 1) > (total_epochs - 25):
    #         with torch.no_grad():
    #             pred_means, pred_variances = pred_means.cpu().numpy(), pred_variances.cpu().numpy()
    #             pred_means, pred_variances = pred_means[0], pred_variances[0]
    #             pred_means = pred_means * y_stats[1] + y_stats[0]
    #             pred_std = np.sqrt(pred_variances) * y_stats[1]
    #             ax_index = ax_counter // 5, ax_counter % 5
    #
    #             ax[ax_index].plot(time[1:], pred_means, label='Predicted data')
    #             ax[ax_index].fill_between(time[1:],
    #                                       pred_means - 2 * pred_std,
    #                                       pred_means + 2 * pred_std,
    #                                       alpha=0.5)
    #             ax[ax_index].scatter(time[1:], measurement, label='Measurement')
    #             ax[ax_index].legend()
    #             ax_counter += 1
    # plt.tight_layout()
    # plt.show()
