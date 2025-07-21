import typing as tp
import numpy as np

from .layers import Layer

class Optimizer:
    """The generic base class for optimizers, used to update trainable
    parameters."""
    def __init__(self, learning_rate: float = 1.0):
        self.learning_rate = learning_rate

    def set_layer_params(self, layers: tp.Iterable[Layer]) -> None:
        """Create a list of dicts corresponding to the sizes of the trainable
        parameters."""
        self.layer_params: tp.Iterable[tp.Dict[str, tp.Tuple[int]]] = []

        for layer in layers:
            # Add trainable param sizes to list
            self.layer_params.append(layer.get_param_dict())

    def update_grad(self, grad: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError()

class SGD(Optimizer):
    """The stochastic gradient descent optimizer only uses the gradient
    and applies it without any momentum terms."""
    def __init__(self, learning_rate: float = 1.0):
        super().__init__(learning_rate)

    def update_grad(self, grad, **kwargs) -> np.ndarray:
        """Returns the adjustment to the parameter based on the gradient. For
        SGD, this is just the gradient multiplied by the learning rate.

        Args:
            grad (_type_): The loss gradient with respect to the parameter.

        Returns:
            np.ndarray: The change in the parameter.
        """
        return self.learning_rate * grad

