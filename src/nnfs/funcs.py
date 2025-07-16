import typing as tp
import numpy as np

class Func:
    def __init__(self):
        pass

    def forward(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def deriv(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class Sigmoid(Func):
    def __init__(self):
        super().__init__()

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-arr))

    def deriv(self, arr: np.ndarray) -> np.ndarray:
        return self.forward(arr) * (1 - self.forward(arr))


class ReLU(Func):
    def __init__(self):
        super().__init__()

    def forward(self, arr: np.ndarray) -> np.ndarray:
        return np.maximum(0, arr)

    def deriv(self, arr: np.ndarray) -> np.ndarray:
        return np.where(arr >= 0, 1, 0)

class SoftMax(Func):
    def __init__(self):
        super().__init__()

    def forward(self, arr: np.ndarray) -> np.ndarray:
        exp_arr = np.exp(arr - np.max(arr, axis = 1, keepdims=True))
        return exp_arr / np.sum(exp_arr, axis=1, keepdims=True)

    def deriv(self, arr: np.ndarray) -> np.ndarray:
        softmax_arr = self.forward(arr)
        return np.diag(softmax_arr) - np.outer(softmax_arr, softmax_arr)


class MSE(Func):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.5 * (y - x)**2

    def deriv(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x - y

