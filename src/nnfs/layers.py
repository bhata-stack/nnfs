import typing as tp
import numpy as np

from .funcs import Func, Sigmoid

class Layer:
    """The base parent class defining the base structure of a Layer object."""
    def __init__(
        self,
        input_shape: tp.Union[tp.Tuple[int], int],
        output_shape: tp.Union[tp.Tuple[int], int],
        **kwargs,
    ):
        """Set generic properties"""
        # Set input and output shape of layer
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Set kwarg options
        self.name: tp.Optional[str] = kwargs.get("name")
        self.batch: bool = kwargs.get("batch", True)

    # Forward and backward propagation are implemented in subclasses
    def forward(self, input_arr: np.ndarray, batching: bool = True) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, output_arr: np.ndarray, batching: bool = True) -> np.ndarray:
        raise NotImplementedError()

    def get_param_dict(self) -> tp.Dict[str, tp.Tuple[int]]:
        """Return a dict of param names and sizes"""
        raise NotImplementedError()

    def reset_params(self) -> None:
        """Reset trainable parameters"""
        raise NotImplementedError()


class LinearLayer(Layer):
    """The basic perceptron layer, defined as y = f(wx + b), where the weight
    and bias (w and b) are trainable and f represents a nonlinear activation
    function."""
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
        self.n_params = self.weight.size + self.bias.size
        self.param_size = self.weight.nbytes + self.bias.nbytes

        # Set activation function
        self.nl_func = activation_func


    def forward(self, input_arr: np.ndarray, batching: bool = True) -> np.ndarray:
        """Given an input array (batched or otherwise), perform a
        forward pass through the layer.

        Args:
            input_arr (np.ndarray): The input for the forward pass.
            batching (bool): Whether or not the input is batched, defaults
            to True.

        Returns:
            np.ndarray: The result of the forward pass.
        """
        # Check if batching, apply weight and bias
        if batching:
            linear_out = np.einsum('ij,kj->ki', self.weight, input_arr) + self.bias[np.newaxis, ...]
        else:
            linear_out = self.weight @ input_arr + self.bias

        # Apply nonlinear activation function
        out = self.nl_func.forward(linear_out)

        # Save input and linear output for backpropagation
        self.l_in = input_arr
        self.l_out = linear_out

        return out

    def backward(self, grad_arr: np.ndarray, **kwargs) -> np.ndarray:
        """Given a gradient array of the output, perform a backward
        pass and optionally update the weight and bias of the layer.

        Args:
            grad_arr (np.ndarray): The loss gradient with respect to the
            output of the layer.
            batching (bool): Whether or not the input is batched, defaults
            to True.
            update_weights (bool): Whether or not to update the weight and
            bias, defaults to True.

        Returns:
            np.ndarray: The loss gradient with respect to the input of the
            layer.
        """
        batching = kwargs.get("batching", True)
        update_weights = kwargs.get("update_weights", True)

        # Derivative of activation function evaluated at output
        deriv = self.nl_func.deriv(self.l_out)
        grad_tot = grad_arr * deriv

        # Calculate gradient in weight and bias
        d_weight = np.einsum('ij,ik->ijk', grad_tot, self.l_in)
        d_bias = grad_tot

        # Calculate gradient over batch
        if batching:
            d_weight = np.mean(d_weight, axis=0)
            d_bias = np.mean(d_bias, axis=0)

        # Use optimizer to calculate change in parameters
        if update_weights:
            self.weight -= self.optimizer.update_grad(d_weight, param = "weight")
            self.bias -= self.optimizer.update_grad(d_bias, param = "bias")

        # Update loss gradient for next layer
        delta_out = grad_tot @ self.weight

        return delta_out

    def get_param_dict(self) -> tp.Dict[str, tp.Tuple[int]]:
        """Return a dict of the trainable parameter shapes for optimization."""
        param_dict = {"weight": self.weight.shape, "bias": self.bias.shape}
        return param_dict

    def reset_params(self) -> None:
        """Reset the weight and bias to random values."""
        # Use existing params to set shape
        self.weight = np.random.standard_normal(self.weight.shape).astype(np.float32)
        self.bias = np.random.standard_normal(self.bias.shape).astype(np.float32)

        return None

