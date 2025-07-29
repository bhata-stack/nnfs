import typing as tp
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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


class Conv2DLayer(Layer):

    def __init__(
        self,
        input_shape: tp.Tuple[int],
        output_shape: tp.Tuple[int],
        filter_shape: tp.Tuple[int],
        activation_func: Func = Sigmoid(),
        **kwargs,
    ):
        # Initialize from BaseLayer
        super().__init__(input_shape, output_shape, **kwargs)

        # Extract kwargs
        self.padding = kwargs.get("padding", False)
        self.pad_value = kwargs.get("pad_value", 0)

        # Validate input and output shapes
        if (len(input_shape) != 2) or (len(output_shape) != 2):
            raise RuntimeError("Both input and output shapes must be 2D.")
        elif len(filter_shape) != 2:
            raise RuntimeError("Filter shape must be 2D.")

        if not self.padding:
            # Calculate output shape
            filter_height, filter_width = filter_shape
            height = input_shape[0] - (filter_height - 1)
            width = input_shape[1] - (filter_width - 1)
            if (height != output_shape[0]) or (width != output_shape[1]):
                raise RuntimeError("Output shape does not match filter output.")
        else:
            # Calculate padding
            self.pad_height = output_shape[0] - input_shape[0] + (filter_shape[0] - 1)
            self.pad_height = int(self.pad_height // 2)
            self.pad_width = output_shape[1] - input_shape[1] + (filter_shape[1] - 1)
            self.pad_width = int(self.pad_width // 2)

        # Calculate padding for backpropagation
        self.back_pad_height = input_shape[0] - output_shape[0] + (filter_shape[0] - 1)
        self.back_pad_height = int(self.back_pad_height // 2)
        self.back_pad_width = input_shape[1] - output_shape[1] + (filter_shape[1] - 1)
        self.back_pad_width = int(self.back_pad_width // 2)

        # Set trainable filter parameter
        self.filter = np.random.standard_normal(filter_shape).astype(np.float32)
        self.n_params = self.filter.size
        self.param_size = self.filter.nbytes

        # Set activation function
        self.nl_func = activation_func


    def forward(self, input_arr: np.ndarray, batching: bool = True) -> np.ndarray:

        arr = input_arr.copy()

        # Check if batching, apply 2D convolution
        if batching:
            # Adjust for padding
            if self.padding:
                pad_shape = (
                    (0, 0),
                    (self.pad_height, self.pad_height),
                    (self.pad_width, self.pad_width)
                )
                arr = np.pad(arr, pad_shape, mode = "constant", constant_values = self.pad_value)

            wind_view = sliding_window_view(arr, self.filter.shape, axis = (1, 2))
            conv_out = np.einsum('ij,hklij->hkl', self.filter, wind_view)
        else:
            if self.padding:
                pad_shape = (
                    (self.pad_height, self.pad_height),
                    (self.pad_width, self.pad_width)
                )
                arr = np.pad(arr, pad_shape, mode = "constant", constant_values = self.pad_value)

            wind_view = sliding_window_view(arr, self.filter.shape)
            conv_out = np.einsum('ij,klij->kl', self.filter, wind_view)

        # Apply nonlinear activation function
        out = self.nl_func.forward(conv_out)

        # Save input and convolution output for backpropagation
        self.in_arr = arr
        self.conv_out = conv_out

        return out


    def backward(self, grad_arr: np.ndarray, **kwargs) -> np.ndarray:


        batching = kwargs.get("batching", True)
        update_weights = kwargs.get("update_weights", True)

        # Derivative of activation function evaluated at output
        local_grad = self.nl_func.deriv(self.conv_out)
        conv_grad = grad_arr * local_grad

        # Calculate gradient over batch
        if batching:
            wind_view = sliding_window_view(self.in_arr, conv_grad.shape[1:], axis = (1, 2))
            filter_grad = np.einsum('hij,hklij->hkl', conv_grad, wind_view)
            filter_grad = np.mean(filter_grad, axis = 0)
        else:
            wind_view = sliding_window_view(self.in_arr, conv_grad.shape)
            filter_grad = np.einsum('ij,klij->kl', conv_grad, wind_view)

        # Use optimizer to calculate change in parameters
        if update_weights:
            self.filter -= self.optimizer.update_grad(filter_grad, param = "filter")

        # Update loss gradient for next layer
        rotated_filter = self.filter[::-1, ::-1]
        # Apply full convolution
        if batching:
            pad_shape = (
                (0, 0),
                (self.back_pad_height, self.back_pad_height),
                (self.back_pad_width, self.back_pad_width)
            )
            pad_conv_grad = np.pad(conv_grad, pad_shape, mode = "constant", constant_values = 0)
            wind_view = sliding_window_view(pad_conv_grad, rotated_filter.shape, axis = (1, 2))
            input_grad = np.einsum('ij,hklij->hkl', rotated_filter, wind_view)
        else:
            pad_shape = (
                (self.back_pad_height, self.back_pad_height),
                (self.back_pad_width, self.back_pad_width)
            )
            pad_conv_grad = np.pad(conv_grad, pad_shape, mode = "constant", constant_values = 0)
            wind_view = sliding_window_view(pad_conv_grad, rotated_filter.shape)
            input_grad = np.einsum('ij,klij->kl', rotated_filter, wind_view)

        return input_grad

    def get_param_dict(self) -> tp.Dict[str, tp.Tuple[int]]:
        """Return a dict of the trainable filter shape for optimization."""
        param_dict = {"filter": self.filter.shape}
        return param_dict

    def reset_params(self) -> None:
        """Reset the filter to random values."""
        # Use existing params to set shape
        self.filter = np.random.standard_normal(self.filter.shape).astype(np.float32)

        return None

class FlattenLayer(Layer):

    def __init__(
        self,
        input_shape: tp.Tuple[int],
        output_shape: tp.Tuple[int],
        **kwargs,
    ):
        # Initialize from BaseLayer
        super().__init__(input_shape, output_shape, **kwargs)

        # Validate input and output shapes
        if isinstance(self.input_shape, int):
            raise RuntimeError("Input is already flattened")
        else:
            input_size = np.multiply.reduce([*self.input_shape])
            if input_size != output_shape:
                raise RuntimeError("Input and output sizes do not match")

        # This layer has no trainable parameters
        self.n_params = 0
        self.param_size = 0

    def forward(self, input_arr: np.ndarray, batching: bool = True) -> np.ndarray:

        # Check if batching
        if batching:
            batch_size = input_arr.shape[0]
            flattened_size = int(input_arr.size / batch_size)
            output_arr = input_arr.reshape(batch_size, flattened_size)
        else:
            output_arr = input_arr.flatten()

        return output_arr

    def backward(self, grad_arr: np.ndarray, **kwargs) -> np.ndarray:

        # Unpack kwargs
        batching = kwargs.get("batching", True)

        # Reshape with batch size
        if batching:
            batch_size = grad_arr.shape[0]
            output_grad = grad_arr.reshape(batch_size, *self.input_shape)
        else:
            output_grad = grad_arr.reshape(*self.input_shape)

        return output_grad

    def get_param_dict(self) -> tp.Dict[str, tp.Tuple[int]]:
        """Return a dict of the trainable parameter shapes for optimization."""
        param_dict = {}
        return param_dict

    def reset_params(self) -> None:
        """Reset trainable parameters."""
        return None