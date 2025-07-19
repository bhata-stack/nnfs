import typing as tp
import numpy as np

from .funcs import Func, Sigmoid

class Layer:
    """PLACEHOLDER"""
    def __init__(
        self,
        input_shape: tp.Union[tp.Tuple[int], int],
        output_shape: tp.Union[tp.Tuple[int], int],
        **kwargs,
    ):
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
        input_shape: tp.Tuple[int],
        output_shape: tp.Tuple[int],
        activation_func: Func = Sigmoid(),
        **kwargs,
    ):
        # Initialize from BaseLayer
        super().__init__(input_shape, output_shape, **kwargs)

        # Determine parameter shapes from input
        weight_shape_1 = output_shape if isinstance(output_shape, int) else output_shape[-1]
        weight_shape_2 = input_shape if isinstance(input_shape, int) else input_shape[-1]
        weight_shape = (weight_shape_1, weight_shape_2)

        bias_shape_2 = output_shape if isinstance(output_shape, int) else output_shape[-1]
        if isinstance(input_shape, int):
            bias_shape = bias_shape_2
        else:
            bias_shape = (*input_shape[:-1], bias_shape_2)

        # Set parameters
        self.weight = np.random.standard_normal(weight_shape).astype(np.float32)
        self.bias = np.random.standard_normal(bias_shape).astype(np.float32)

        self.nl_func = activation_func


    def forward(self, input_arr: np.ndarray) -> np.ndarray:

        # Check if batching, apply weight and bias
        if input_arr.ndim == 1:
            batching = input_arr.shape[0] == self.input_shape
        else:
            batching = input_arr.shape == self.input_shape
        if batching:
            linear_out = self.weight @ input_arr + self.bias
        else:
            linear_out = np.einsum('ij,kj->ki', self.weight, input_arr) + self.bias[np.newaxis, ...]

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
        deriv = self.nl_func.deriv(self.l_out)
        delta_tot = delta_arr * deriv

        # Calculate gradient in weight and bias
        d_weight = np.einsum('ij,ik->ijk', delta_tot, self.l_in)
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

        delta_out = delta_tot @ self.weight

        return delta_out
