import typing as tp
import numpy as np

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
            self.params[key] = np.random.rand(*value) if isinstance(value, tuple) else value

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
