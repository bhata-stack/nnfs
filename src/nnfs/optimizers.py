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

        return None

    def update_grad(self, grad: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def reset_optimizer(self) -> None:
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

    def reset_optimizer(self) -> None:
        return None


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
    ):
        # Initialize with learning rate
        super().__init__(learning_rate)

        # Set other constants
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self.t_step = 1

    def set_layer_params(self, layers: tp.Iterable[Layer]) -> None:

        # Set layer_params property
        super().set_layer_params(layers)

        # Use layer_params to initialize first and second moments
        self.first_moment = []
        self.second_moment = []
        for params in self.layer_params:
            param_dict = {key: np.zeros(shape) for key, shape in params.items()}
            self.first_moment.append(param_dict.copy())
            self.second_moment.append(param_dict)


    def update_grad(self, grad: np.ndarray, param: str, layer_num: int, **kwargs) -> np.ndarray:
        """Returns the adjustment to the parameter based on the gradient. For
        SGD, this is just the gradient multiplied by the learning rate.

        Args:
            grad (_type_): The loss gradient with respect to the parameter.

        Returns:
            np.ndarray: The change in the parameter.
        """

        # Extract first and second moments
        first_moment = self.first_moment[layer_num][param]
        second_moment = self.second_moment[layer_num][param]

        # Update moments, calculate adjusted moments for param update
        first_moment = self.beta_1 * first_moment + (1 - self.beta_1) * grad
        second_moment = self.beta_2 * second_moment + (1 - self.beta_2) * grad**2

        fm_hat = first_moment / (1 - self.beta_1 ** self.t_step)
        sm_hat = second_moment / (1 - self.beta_2 ** self.t_step)

        param_update = self.learning_rate * fm_hat / (np.sqrt(sm_hat) + self.eps)

        # Update time step
        if layer_num == len(self.first_moment) - 1:
            self.t_step += 1

        return param_update

    def reset_optimizer(self, t_step_only: bool = False) -> None:

        if not t_step_only:
            for fm, sm in zip(self.first_moment, self.second_moment):
                fm = {param: np.zeros_like(arr) for param, arr in fm.items()}
                sm = {param: np.zeros_like(arr) for param, arr in sm.items()}

        self.t_step = 1

        return None


