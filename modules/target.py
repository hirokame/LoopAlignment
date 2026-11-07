import numpy as np

# ターゲットネットワーク（固定）
class TargetNetwork:
    def __init__(self, dims=[64, 8, 32, 64]):
        self.weights = [
            np.random.uniform(low=-0.5, high=0.5, size=(dims[i], dims[i+1]))
            for i in range(len(dims)-1)
        ]
        self.biases = [
            np.random.uniform(low=-0.5, high=0.5, size=(1, dims[i+1]))
            for i in range(len(dims)-1)
        ]

    def forward(self, x):
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ W + b)
        return x @ self.weights[-1] + self.biases[-1]  # 最後は線形出力
