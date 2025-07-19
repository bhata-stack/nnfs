"""PLACEHOLDER"""
import typing as tp
from time import monotonic

import numpy as np
from tqdm import tqdm

from .layers import Layer
from . import funcs

class BaseModel:
    """PLACEHOLDER"""
    def __init__(self, layers: tp.List[Layer], opt_func: tp.Callable[np.ndarray, np.ndarray]):

        self.layers = layers

        self.n_params = 0

        pass

    def forward(self, input_arr: np.ndarray) -> np.ndarray:

        arr = input_arr.copy()

        for layer in self.layers:
            arr = layer.forward(arr)

        return arr

    def backward(self, delta_arr: np.ndarray, **kwargs):

        batching = kwargs.get("batching", True)
        update_weights = kwargs.get("update_weights", True)

        arr = delta_arr.copy()

        for layer in reversed(self.layers):
            arr = layer.backward(arr, update_weights = update_weights)

        return arr

    def train(
        self,
        training_data: tp.Iterable[np.ndarray],
        testing_data: tp.Iterable[np.ndarray],
        batch_size: int = 32,
        epochs: int = 5,
        loss_fn: funcs.Func = funcs.CrossEntropy(),
        optimizer: funcs.Optimizer = None,
    ):

        # Start timing
        start_time = monotonic()

        # Extract arrays
        x_train, y_train = training_data
        x_test, y_test = testing_data

        # Determine number of batches in each epoch
        n_batches = x_train.shape[0] // batch_size

        for epoch_ind in range(epochs):
            print(f"Start of Epoch {epoch_ind + 1}:")

            # Shuffle data
            n_train_samples = x_train.shape[0]
            n_test_samples = x_test.shape[0]

            rand_ind = np.random.choice(n_train_samples, n_train_samples, replace = False)
            x_train, y_train = x_train[rand_ind], y_train[rand_ind]

            rand_ind = np.random.choice(n_test_samples, n_test_samples, replace = False)
            x_test, y_test = x_test[rand_ind], y_test[rand_ind]

            # Loop over batches
            for n in tqdm(range(n_batches)):
                x_train_batch = x_train[n * batch_size:(n + 1) * batch_size]
                y_train_batch = y_train[n * batch_size:(n + 1) * batch_size]

                # Forward pass
                output = self.forward(x_train_batch)

                # Calculate gradient
                output_grad = loss_fn.deriv(output, y_train_batch)

                # Update weights
                input_grad = self.backward(output_grad)

            # Print loss on testing set
            rand_ind = np.random.randint(0, x_test.shape[0], batch_size)
            test_output = self.forward(x_test[rand_ind])
            loss = loss_fn.forward(test_output, y_test[rand_ind])
            print(f"Average loss at epoch {epoch_ind + 1}: {np.mean(loss):.5f}")

        # Print training time
        time_taken = monotonic() - start_time
        hour = time_taken // 3600
        minute = (time_taken % 3600) // 60
        second = time_taken % 60
        print(f"Training took {hour:2d}h {minute:2d}m {second:2d}s.")
