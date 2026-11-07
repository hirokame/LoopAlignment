import numpy as np

class Func:
    def __init__(self, name, f, f_deriv, limit, b):
        self.name = name
        self.f = f
        self.f_deriv = f_deriv
        self.limit = limit
        self.b = b

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return self.f(x)

    def deriv(self, x:np.ndarray) -> np.ndarray:
        return self.f_deriv(x)

    def init_weight(self, x_dim:int, y_dim:int) -> tuple[float, np.ndarray]:
        return self.limit(x_dim, y_dim), self.b(y_dim)
    
    def get_name(self) -> str:
        return self.name

LINEAR = Func(
    "LINEAR",
    lambda x: x,
    lambda x: np.ones(x.shape),
    lambda xd,yd: np.sqrt(1/(xd+yd)),
    lambda yd: np.zeros((1, yd))
)
TANH = Func(
    "TANH",
    lambda x: np.tanh(x),
    lambda x: np.ones_like(x) - np.tanh(x) ** 2,
    lambda xd,yd: np.sqrt(6/(xd+yd)),
    lambda yd: np.zeros((1, yd))
)
RELU = Func(
    "RELU",
    lambda x: x.clip(min=0),
    lambda x: x.clip(min=0),
    lambda xd,yd: np.sqrt(1/(xd+yd)),
    lambda yd: np.ones((1, yd))*0.1
)
LOGSIG = Func(
    "LOGSIG",
    lambda x: 1. / (np.ones_like(x) + np.exp(-x)),
    lambda x: np.multiply(1/(np.ones_like(x) + np.exp(-x)), (np.ones_like(x) - (1/ (np.ones_like(x) + np.exp(-x))))),
    lambda xd,yd: 4*np.sqrt(6/(xd+yd)),
    lambda yd: np.ones((1, yd))*0.1
)