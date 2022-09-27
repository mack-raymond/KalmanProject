import numpy as np


def generate_exponential_data(ammount, remove_block=0):
    t = np.linspace(-1 * np.random.rand(), np.random.rand() * 30, ammount + remove_block)
    data = np.exp((t * np.random.rand() + np.random.randn()) * np.random.rand() * 10) + np.random.randn() * 10
    t[np.random.randint(ammount):] += np.abs(np.random.randn()) * 10
    data[np.random.randint(ammount):] += np.random.randn() * 10
    indices_to_remove = np.random.randint(0, ammount - remove_block)
    t = np.delete(t, np.arange(indices_to_remove, indices_to_remove + remove_block))
    data = np.delete(data, np.arange(indices_to_remove, indices_to_remove + remove_block))
    noise = np.random.randn(ammount) * 10
    return t, data + noise


def generate_cubic_data(ammount, remove_block=0):
    t = np.linspace(-10 * np.random.rand(), np.random.rand() * 20, ammount + remove_block)
    data = (t * np.random.rand() + np.random.randn() * 10) ** 3 + np.random.randn() * 10
    t[np.random.randint(ammount):] += np.abs(np.random.randn()) * 10
    data[np.random.randint(ammount):] += np.random.randn() * 10
    indices_to_remove = np.random.randint(0, ammount - remove_block)
    t = np.delete(t, np.arange(indices_to_remove, indices_to_remove + remove_block))
    data = np.delete(data, np.arange(indices_to_remove, indices_to_remove + remove_block))
    noise = np.random.randn(ammount) * 10
    return t, data + noise


def generate_sin_data(ammount, remove_block=0):
    t = np.linspace(-10 * np.random.rand(), np.random.rand() * 20, ammount + remove_block)
    data = np.sin(t * np.random.rand() + np.random.randn()) + np.random.randn() * 10
    t[np.random.randint(ammount):] += np.abs(np.random.randn()) * 10
    data[np.random.randint(ammount):] += np.random.randn() * 10
    indices_to_remove = np.random.randint(0, ammount - remove_block)
    t = np.delete(t, np.arange(indices_to_remove, indices_to_remove + remove_block))
    data = np.delete(data, np.arange(indices_to_remove, indices_to_remove + remove_block))
    noise = np.random.randn(ammount) * 10
    return t, data + noise
