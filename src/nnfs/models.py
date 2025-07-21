import typing as tp
from time import monotonic

import numpy as np
from tqdm import tqdm

from .layers import Layer
from . import funcs
from .optimizers import Optimizer, SGD

class BaseModel:
    """The base neural network model, defined as a list of layers that perform
    certain operations and can optionally be trained."""

    def __init__(
        self,
        layers: tp.List[Layer],
        optimizer: Optimizer = SGD(),
        **kwargs
    ):
        """Initialize the model with the provided layers and optimizer.

        Args:
            layers (tp.List[Layer]): A list of the layers in the order
            of operations.
            optimizer (Optimizer, optional): The optimizer to use for
            updating trainable weights. Defaults to Stochastic Gradient
            Descent.
            print_out (bool): Whether or not to print the model size during
            initialization.
        """
        # Process kwargs
        print_out = kwargs.get("print_out", True)

        # Initialize layers and optimizer
        self.layers = layers
        self.set_optimizer(optimizer)

        # Set layer number, optimizer for each layer
        for i, layer in enumerate(self.layers):
            layer.layer_num = i
            layer.optimizer = self.optimizer

        # Get size of model
        self.n_params = sum([layer.n_params for layer in layers])
        self.param_size = sum([layer.param_size for layer in layers])

        # Print model size
        if print_out:
            size_gb = self.param_size / (1024**3)
            if size_gb > 1:
                size = f"{size_gb:.1f} GB"
            else:
                size = f"{self.param_size / (1024**2):.1f} MB"
            print(f"Initialized model with {self.n_params} parameters ({size}).")


    def forward(self, input_arr: np.ndarray, batching: bool = True) -> np.ndarray:
        """Given an input array (batched or otherwise), perform a
        forward pass through the model. For trained models, this is
        essentially the same as an evaluation.

        Args:
            input_arr (np.ndarray): The input for the forward pass.
            batching (bool): Whether or not the input is batched, defaults
            to True.

        Returns:
            np.ndarray: The model output.
        """
        # Copy input as to not overwrite the original array
        arr = input_arr.copy()

        # Forward pass through the layers
        for layer in self.layers:
            arr = layer.forward(arr, batching = batching)

        return arr

    def backward(self, delta_arr: np.ndarray, **kwargs) -> np.ndarray:
        """Given a loss gradient, perform backward propagation through
        the model, optionally updating the weights as to minimize the
        loss.

        Args:
            delta_arr (np.ndarray): The loss gradient with respect to the
            model output.
            batching (bool): Whether or not the input is batched, defaults
            to True.
            update_weights (bool): Whether or not to update the weight and
            bias, defaults to True.

        Returns:
            np.ndarray: The loss gradient with respect to the original model
            input.
        """
        batching = kwargs.get("batching", True)
        update_weights = kwargs.get("update_weights", True)

        arr = delta_arr.copy()

        for layer in reversed(self.layers):
            arr = layer.backward(arr, batching = batching, update_weights = update_weights)

        return arr

    def train(
        self,
        training_data: tp.Iterable[np.ndarray],
        testing_data: tp.Iterable[np.ndarray],
        loss_fn: funcs.Func,
        **kwargs,
    ) -> None:
        """Train the model using both training and testing data for a specified
        number of epochs, printing the loss after each epoch. The trainable
        weights for each layer are updated with each batch of training data.

        Args:
            training_data (tp.Iterable[np.ndarray]): An array containing
            training samples, with the first axis corresponding to the
            number of samples.
            testing_data (tp.Iterable[np.ndarray]): An array containing samples
            to be used for testing, with the first axis corresponding to the
            number of samples.
            loss_fn (funcs.Func, optional): The function used to calculate the
            output loss of the model.
            batch_size (int): The size of batches used for calculating gradients,
            defaults to 32.
            epochs (int): The number of epochs to train over, defaults to 5.
            full_test (bool): Whether or not to use the full test data for
            the loss calculation at the end of each epoch, defaults to True.

        Returns:
            None
        """
        # Extract kwargs
        batch_size = kwargs.get("batch_size", 32)
        epochs = kwargs.get("epochs", 5)
        full_test = kwargs.get("full_test", False)

        # Extract arrays
        x_train, y_train = training_data
        x_test, y_test = testing_data

        # Determine number of batches in each epoch
        n_batches = x_train.shape[0] // batch_size

        # Start timing
        start_time = monotonic()

        # Loop over each epoch
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
            if full_test:
                test_output = self.forward(x_test)
                loss = loss_fn.forward(test_output, y_test)
            else:
                rand_ind = np.random.randint(0, x_test.shape[0], batch_size)
                test_output = self.forward(x_test[rand_ind])
                loss = loss_fn.forward(test_output, y_test[rand_ind])

            print(f"Average loss at epoch {epoch_ind + 1}: {np.mean(loss):.5f}")

        # Print training time
        time_taken = monotonic() - start_time
        hour = int(time_taken // 3600)
        minute = int((time_taken % 3600) // 60)
        second = int(time_taken % 60)
        print(f"Training took {hour:2d}h {minute:2d}m {second:2d}s.")

        return None


    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer used for updating trainable parameters."""
        # Assign optimizer
        self.optimizer = optimizer

        # Get trainable parameters from each layer
        self.optimizer.set_layer_params(self.layers)

        return None


    def reset_params(self) -> None:
        """Reset all parameters of the model."""
        for layer in self.layers:
            layer.reset_params()

        return None
