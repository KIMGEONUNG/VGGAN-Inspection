import numpy as np


class GrayConversion():

    def __init__(self,
                 name="classic",
                 preprocess=lambda x: x,
                 postprocess=lambda x: x):
        self.fn_defualt = getattr(self, name)
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.w_classic = np.array([0.299, 0.587, 0.114])

    def gen_method3str(self, x, name):
        x = self.preprocess(x)
        x = getattr(self, name)(x)
        x = self.postprocess(x)
        return x

    def heavy(self, x):
        w = np.random.randn(3)
        x = x * 2 - 1
        x = (x @ w)[..., None]
        x = np.tanh(x)
        x = (x + 1) * 0.5
        return x

    def scale(self, x):
        g = (x @ self.w_classic)[..., None]

        shrink = np.random.uniform(0, 1)
        bias = np.random.uniform(0, 1)

        shrink = max(shrink, 0.1)
        bias = min(bias, 1.0 - shrink)
        g = g * shrink + bias

        return g

    def classic(self, x):
        return (x @ self.w_classic)[..., None]

    def __call__(self, x):
        x = self.preprocess(x)
        x = self.fn_defualt(x)
        x = self.postprocess(x)
        return x
