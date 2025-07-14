"""PLACEHOLDER"""
import typing as tp
import numpy as np
from layers import Layer

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

    def backward(self):

        pass