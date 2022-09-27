import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from data_generators import generate_exponential_data, generate_cubic_data, generate_sin_data


class DeepSmoother(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                     nn.Linear(hidden_size, 2))

    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        x = self.encoder(x)
        x, _ = self.gru(x)
        x = self.decoder(x)
        smoothing_factor, variances = torch.sigmoid(x[:, :, 0]), torch.exp(x[:, :, 1])
        return smoothing_factor, variances


def normalize_input(x, y, split) -> (np.ndarray, np.ndarray, [np.ndarray, np.ndarray], [np.ndarray, np.ndarray]):
    split = int(split * len(x))
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


if __name__ == '__main__':
    generated_points = 10000
    total_epochs = 100000
    stats_split = 1

    model = DeepSmoother(128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.GaussianNLLLoss(reduction='mean', eps=1e-10)
    model.cuda()

    model.train()
    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    ax_counter = 0
    for epoch in range(total_epochs):
        generate = np.random.choice([generate_cubic_data, generate_exponential_data, generate_sin_data])
        sampled_points = np.random.randint(10, 2000, 1)[0]
        time, measurement = generate(generated_points, np.random.randint(0, 1000, 1)[0])
        indices = np.random.choice(generated_points, sampled_points, replace=False)
        indices = np.sort(indices)
        time, measurement = time[indices], measurement[indices]

        dt = time[1:] - time[:-1]
        initial_measurement = measurement[:-1]
        measurement = measurement[1:]
        x_train = np.stack([initial_measurement, dt], axis=1)
        y_train = measurement.reshape(-1, 1)

        x_train, y_train, x_stats, y_stats = normalize_input(x_train, y_train, stats_split)
        x_train, y_train = x_train.reshape(1, -1, 2), y_train.reshape(1, -1, 1)
        x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()

        x_train, y_train = x_train.cuda(), y_train.cuda()
        pred_means, pred_variances = model(x_train)
        y_train = y_train[:, :, 0]
        loss = loss_fn(pred_means, y_train,
                       pred_variances)
               # pred_variances.mean()
        # ((pred_means - y_train) ** 2).mean() + \
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            optimizer.step()

        print(f'Epoch {epoch} loss: {loss.item()}')
        if (epoch + 1) > (total_epochs - 25):
            with torch.no_grad():
                pred_means, pred_variances = pred_means.cpu().numpy(), pred_variances.cpu().numpy()
                pred_means, pred_variances = pred_means[0], pred_variances[0]
                pred_means = pred_means * y_stats[1] + y_stats[0]
                pred_std = np.sqrt(pred_variances) * y_stats[1]
                ax_index = ax_counter // 5, ax_counter % 5

                ax[ax_index].plot(time[1:], pred_means, label='Predicted data')
                ax[ax_index].fill_between(time[1:],
                                          pred_means - 2 * pred_std,
                                          pred_means + 2 * pred_std,
                                          alpha=0.5)
                ax[ax_index].scatter(time[1:], measurement, label='Measurement')
                ax[ax_index].legend()
                ax_counter += 1
    plt.tight_layout()
    plt.show()
