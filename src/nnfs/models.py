"""PLACEHOLDER"""
import typing as tp
import numpy as np
from .layers import Layer
from . import funcs

class BaseModel:
    """PLACEHOLDER"""
    def __init__(self, layers: tp.List[Layer], opt_func: tp.Callable[np.ndarray, np.ndarray]):

        self.layers = layers

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

        for layer in self.layers:
            arr = layer.backward(arr, update_weights = update_weights)

        return arr

    def train(
        self,
        training_data: np.ndarray,
        testing_data: np.ndarray,
        batch_size: int = 32,
        epochs: int = 5,
        loss_fn: funcs.Func = funcs.MSE,
        optimizer: funcs.Optimizer = None,
    ):

        for epoch_ind in range(epochs):
            # Shuffle data
            np.random.Generator.shuffle(training_data, axis = 1)
            np.random.Generator.shuffle(testing_data, axis = 1)

            # Extract arrays
            x_train, y_train = training_data
            x_test, y_test = testing_data

            # Loop over batches
            n_batches = x_train.shape[0] // batch_size

            for n in range(n_batches):
                x_train_batch = x_train[n * batch_size:(n + 1) * batch_size]
                y_train_batch = y_train[n * batch_size:(n + 1) * batch_size]


                output = self.forward(x_train_batch)
                # Calculate gradient
                output_grad = loss_fn.deriv(output, y_train_batch)

                # Update weights
                input_grad = self.backward(output_grad)

            # Print loss on testing set
            rand_ind = np.random.randint(0, x_test.shape[0], batch_size)
            test_output = self.forward(x_test[rand_ind])
            loss = loss_fn.forward(test_output, y_test[rand_ind])
            print(f"Loss at epoch {epoch_ind}: {loss:.5f}")
