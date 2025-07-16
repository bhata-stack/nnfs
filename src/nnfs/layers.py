import typing as tp
import numpy as np

from .funcs import ActivationFunc, Sigmoid

class Layer:
    """PLACEHOLDER"""
    def __init__(
        self,
        params: tp.Dict[str, tp.Union[tp.Tuple[int], np.ndarray]],
        input_shape: tp.Tuple[int],
        output_shape: tp.Tuple[int],
        **kwargs,
    ):
        # Initialize params if not given array directly
        self.params = {}
        for key, value in params.items():
            if isinstance(value, tuple):
                range_val = 1 / np.sqrt(np.sum(value))
            else:
                self.params[key] = value
            self.params[key] = np.random.uniform(-range_val, range_val, value)

        # Set input and output shape of layer
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Set kwarg options
        self.name: tp.Optional[str] = kwargs.get("name")
        self.batch: bool = kwargs.get("batch", True)

    # Forward and backward propagation are implemented in subclasses
    def forward(self, input_arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, output_arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class LinearLayer(Layer):

    def __init__(
        self,
        params: tp.Dict[str, tp.Union[tp.Tuple[int], np.ndarray]],
        input_shape: tp.Tuple[int],
        output_shape: tp.Tuple[int],
        activation_func: ActivationFunc = Sigmoid,
        **kwargs,
    ):
        # Initialize from BaseLayer
        super().__init__(params, input_shape, output_shape, **kwargs)

        # Set parameters
        self.weight = params["weight"]
        self.bias = params["bias"]

        # Check that input and output shape match with params shape
        weight_shape = (input_shape[-1], output_shape[-1])
        if len(input_shape == 1):
            bias_shape = (output_shape[-1])
        else:
            bias_shape = (*input_shape[:-1], output_shape[-1])
        if weight_shape != self.weight.shape or bias_shape != self.bias.shape:
            raise RuntimeError("Input and Output shapes do not match parameter shapes.")

        self.nl_func = activation_func


    def forward(self, input_arr: np.ndarray) -> np.ndarray:

        # Check if batching, apply weight and bias
        if input_arr.shape == self.input_shape:
            linear_out = input_arr @ self.weight + self.bias
        else:
            linear_out = input_arr @ self.weight + self.bias[np.newaxis, ...]

        # Apply nonlinear activation function
        out = self.nl_func.forward(linear_out)

        # Save input and linear output for backpropagation
        self.l_in = input_arr
        self.l_out = linear_out

        return out

    def backward(self, delta_arr: np.ndarray, **kwargs) -> np.ndarray:

        batching = kwargs.get("batching", True)
        update_weights = kwargs.get("update_weights", True)

        # Derivative of activation function evaluated at output
        deriv = self.nl_func.deriv(self.nl_out)
        delta_tot = delta_arr * deriv

        # Calculate gradient in weight and bias
        d_weight = np.multiply.outer(self.l_in, delta_tot)
        d_bias = delta_tot

        # Calculate gradient over batch
        if batching:
            d_weight = np.mean(d_weight, axis=0)
            d_bias = np.mean(d_bias, axis=0)

        # Use optimizer to calculate change in parameters
        if update_weights:
            # TODO: Implement various optimizers, currently uses SGD
            eta = 0.5
            self.weight -= eta * d_weight
            self.bias -= eta * d_bias

        delta_out = self.weight @ delta_tot

        return delta_out


