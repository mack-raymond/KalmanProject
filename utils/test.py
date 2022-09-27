from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from kalman_shit import KalmanFilter
import time


def transition_function(dt, degree=3):
    F = []
    for row in range(degree):
        if row == 0:
            F.append(np.power(dt, np.arange(degree)) / np.clip(np.arange(degree), 1, None))
        else:
            row_ = np.zeros(degree)
            row_[1:] = F[row - 1][:-1]
            F.append(row_)
    return np.array(F)


def true_function(points):
    F = transition_function(1)
    x = np.array([0, 0, 0.01]).reshape(-1, 1)
    items = []
    for _ in range(points):
        x = F @ x
        items.append(x[0, 0])

    items = np.array(items)
    # items = np.where(np.arange(points) > points / 2,
    #                  np.random.randn(*x.shape) * 10 / (np.arange(points) - points / 2) + items,
    #                  np.random.randn(*x.shape) * 0 / (np.arange(points)) + items)
    noise = np.concatenate([4000 * np.random.randn(len(items) // 2) / (0.0001 * np.arange(len(items) // 2) + 1),
                            4000 * np.random.randn(len(items) // 2) / (0.0001 * np.arange(len(items) // 2) + 1)])
    items += noise

    # noise = np.where(x > 150, np.random.randn(*x.shape) * (1000 / (1 + 0.1 * (x-150))), np.random.randn(*x.shape) * (1000 / (1 + 0.1 * x)))
    return items.reshape(-1, 1)


x_train = np.arange(20)[:, None] / 1
x_pred = np.concatenate([x_train, np.arange(20, 40)[:, None] / 1])
x_pred = (x_pred - x_train.mean()) / x_train.std()
x_train = (x_train - x_train.mean()) / x_train.std()

y_train = true_function(2000)
y_train = y_train[::100]
y_train = (y_train - y_train.mean()) / y_train.std()

# x_train = (x_train - x_train.mean()) / x_train.std()
# y_train = (y_train - np.mean(y_train)) / np.std(y_train)
degree = 3
variance_smoothing = 0.1

time_start = time.time()
predictions = [y_train[0, 0]]
variances = [np.std(y_train) ** 2]
# meas_var = np.std(y_train) ** 2
# print(meas_var)
meas_var = np.std(y_train) ** 2
meas_vars = [meas_var]
# process_var = np.std(
#     (y_train[1:] - y_train[:-1]) / (((x_train[1:] - x_train[:-1]) ** (degree - 1)) / np.prod(np.arange(1, degree)))) ** 2
# print(y_train[1:] - y_train[:-1])
# print(x_train[1:] - x_train[:-1])
# process_var = 10
# meas_var = 100
# print(process_var)
process_var = 0.00
# process_var = 20 ** 2
# meas_var = 10000 ** 2
dt = (x_train[1] - x_train[0]).item()
kalman_filter = KalmanFilter(initial=y_train[0, 0],
                             initial_uncertainty=2**2,
                             process_variance=process_var,
                             measurement_variance=meas_var,
                             )
# window = 20
# Rs = []
# for i in range(len(y_train) - window):
#     Rs.append(np.std(y_train[i:i + window]) ** 2)
# while len(Rs) < len(y_train):
#     Rs.append(Rs[-1])
# Rs = np.array(Rs)

for step in range(1, len(x_train)):
    x, P, _ = kalman_filter.predict(dt=dt)
    # print(_[0, 0])
    # process_var = (((y_train[step] - x[0, 0]) / (dt ** 2 * 0.5)) - x[-1, 0]).item() ** 2 * (1-variance_smoothing) * 0.001 + process_var * variance_smoothing
    print(meas_var, process_var)
    meas_var = (y_train[step] - kalman_filter.x[0, 0]).item() ** 2 * (1 - variance_smoothing) + meas_var * variance_smoothing

    x, P = kalman_filter.update(measurement=y_train[step],
                                dt=dt,
                                measurement_variance=meas_var,
                                process_variance=process_var)

    predictions.append(x[0, 0])
    variances.append(P[0, 0])
    meas_vars.append(meas_var)

count = 0
while len(predictions) < len(x_pred):
    count += 1
    x, P, _ = kalman_filter.predict(dt=dt * count)
    predictions.append(x[0, 0])
    variances.append(P[0, 0])
    meas_vars.append(meas_var)
predictions = np.array(predictions)
variances = np.array(variances)
meas_vars = np.array(meas_vars)

x_train = x_train[:, 0]
# fig, ax = plt.subplots(1, figsize=(15, 15))
# ax[0].plot(x_pred, predictions, label='Kalman', alpha=0.6)
# print(time.time() - time_start)
# time_start = time.time()
# ax[0].fill_between(x_pred[:, 0],
#                    np.array(predictions) - 2 * np.sqrt(variances),
#                    np.array(predictions) + 2 * np.sqrt(variances),
#                    alpha=0.2)
# ax[0].scatter(x_train, y_train, label='True', s=1)
# ax[0].set_ylim(-500, 10000)
# ax[0].set_title("Kalman Filter")
plt.plot(x_pred, predictions, label='Kalman', alpha=0.6)
plt.fill_between(x_pred[:, 0],
                    np.array(predictions) - 2 * np.sqrt(variances) - 2 * np.sqrt(meas_vars),
                    np.array(predictions) + 2 * np.sqrt(variances) + 2 * np.sqrt(meas_vars),
                    alpha=0.2)
plt.ylim(-10, 10)
plt.scatter(x_train, y_train, label='True', s=1)
plt.tight_layout()
plt.figure(figsize=(15, 15))
plt.show()
exit()

# print(x_train.shape, y_train.shape)
#
# #
# #
# X = np.linspace(x_train[0, 0], x_train[-1, 0] + 10, 1000)[:, None]
#
# # plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
# plt.scatter(x_train, y_train, label="Observations")
#
kernel = 1 * RBF() + WhiteKernel()
# kernel = 1 * RBF() + WhiteKernel(noise_level_bounds=(1e-10, np.std(y_train) ** 2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True, alpha=1)
gaussian_process.fit(x_train.reshape(-1, 1), y_train)
mean_prediction, std_prediction = gaussian_process.predict(x_pred.reshape(-1, 1), return_std=True)
ax[1].plot(x_pred[:, 0], mean_prediction, label='Gaussian Process', alpha=0.6)
print(time.time() - time_start)
# mean_prediction = mean_prediction
print(mean_prediction.shape, std_prediction.shape)
# plt.plot(X[:, 0], mean_prediction, label="Mean prediction")
ax[1].fill_between(
    x_pred[:, 0],
    mean_prediction - 2 * std_prediction,
    mean_prediction + 2 * std_prediction,
    alpha=0.2,
)


# # plt.legend()
# # plt.scatter(x_train, y_train, label="Observations")
# # plt.xlabel("$x$")
# # plt.ylabel("$f(x)$")
# # plt.show()
#
class UncertaintyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64),
                                   # nn.Dropout(0.001),
                                   # nn.BatchNorm1d(64),
                                   nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        out = self.model(x)
        mean = out[:, 0, None]
        std = torch.clamp(out[:, 1, None], min=1e-5)
        return torch.distributions.Normal(mean, std)


dl_model = UncertaintyModel()
x_train_torch = torch.from_numpy(x_train).float().reshape(-1, 1)
y_train_torch = torch.from_numpy(y_train).float().reshape(-1, 1)
x_train_torch_mean = torch.mean(x_train_torch)
x_train_torch_std = torch.std(x_train_torch)
y_train_torch_mean = torch.mean(y_train_torch)
y_train_torch_std = torch.std(y_train_torch)
x_train_torch = (x_train_torch - x_train_torch_mean) / x_train_torch_std
y_train_torch = (y_train_torch - y_train_torch_mean) / y_train_torch_std
optimizer = torch.optim.AdamW(dl_model.parameters(), lr=1e-3, weight_decay=1e-4)
for _ in range(10000):
    optimizer.zero_grad()
    loss = -dl_model(x_train_torch).log_prob(y_train_torch).mean()
    # #     y_pred = dl_model(torch.from_numpy(x_train).float() + torch.randn(*x_train.shape)*0.001)
    # #     # loss = loss_fn(y_pred[:, 0] + torch.randn(*y_pred[:, 0].shape)*2 * y_pred[:, 1], torch.from_numpy(y_train).float()[:, 0])
    # #     # loss = ((y_pred[:, 0] - torch.from_numpy(y_train).float()[:, 0]).pow(2) / y_pred[:, 1].pow(2)).mean() + y_pred[:, 1].mean()*0.0001
    # #     loss = (torch.log((y_pred[:, 0] - torch.from_numpy(y_train).float()[:, 0]).pow(2)) - torch.log(y_pred[:, 1].pow(2))).mean()
    # #     # loss = torch.log((y_pred[:, 0]-torch.from_numpy(y_train).float()[:, 0]).pow(2).mean()) - torch.log(y_pred[:, 1])).mean()
    loss.backward()
    optimizer.step()
    print(loss.item())
# #
with torch.no_grad():
    dl_model.eval()
    x_pred_torch = torch.from_numpy(x_pred).float()
    x_pred_torch = (x_pred_torch - x_train_torch_mean) / x_train_torch_std
    y_pred = dl_model(x_pred_torch)
# # y_pred = dl_model(torch.from_numpy(X).float())
dl_y_mean, dl_y_std = y_pred.mean[:, 0], y_pred.stddev[:, 0]
dl_y_mean = dl_y_mean * y_train_torch_std + y_train_torch_mean
dl_y_std = dl_y_std * y_train_torch_std
ax[2].plot(x_pred[:, 0], dl_y_mean, label='Deep Learning', alpha=0.6)
ax[2].fill_between(
    x_pred[:, 0],
    dl_y_mean - 2 * dl_y_std,
    dl_y_mean + 2 * dl_y_std,
    alpha=0.2,
)
ax[2].set_title('Deep Learning')
# plt.plot(X, dl_y_mean, label="Mean prediction")
# plt.fill_between(
#     X.ravel(),
#     dl_y_mean - 1.96 * dl_y_std,
#     dl_y_mean + 1.96 * dl_y_std,
#     alpha=0.5,
#     label=r"95% confidence interval",
#     )
# plt.show()
ax[1].scatter(x_train, y_train, label='True', s=1)
ax[2].scatter(x_train, y_train, label='True', s=1)
ax[1].set_ylim(-500, 10000)
ax[2].set_ylim(-500, 10000)
ax[1].set_title("Gaussian Process")
# plt.ylim(-5000, 40000)
# plt.legend()
# plt.show()
ax[0].scatter(x_train, y_train, label='True', s=1)
ax[0].set_ylim(-500, 10000)
ax[0].set_title("Kalman Filter")
plt.tight_layout()
fig.show()
