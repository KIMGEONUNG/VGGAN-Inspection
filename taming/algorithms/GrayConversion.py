import numpy as np
import random
from taming.algorithms.SignalProcessor import SignalProcessor


class GrayConversion():

    def __init__(self,
                 name="classic",
                 preprocess=SignalProcessor.identity,
                 postprocess=SignalProcessor.identity):
        self.fn_defualt = getattr(self, name)
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.w_classic = np.array([0.299, 0.587, 0.114])

    def gen_method3str(self, x, name):
        x = self.preprocess(x)
        x, named_param = getattr(self, name)(x)
        x = self.postprocess(x)
        return x, named_param

    def normal_weight_A(self, x):
        w = np.random.randn(3)
        x = x * 2 - 1
        x = (x @ w)[..., None]
        x = np.tanh(x)
        x = (x + 1) * 0.5

        # Params
        named_params = {}
        named_params["w_R"] = w[0]
        named_params["w_G"] = w[1]
        named_params["w_B"] = w[2]

        return x, named_params

    def normal_weight_A_equalsign(self, x):
        w = self._random_sign() * abs(np.random.randn(3))
        x = x * 2 - 1
        x = (x @ w)[..., None]
        x = np.tanh(x)
        x = (x + 1) * 0.5

        # Params
        named_params = {}
        named_params["w_R"] = w[0]
        named_params["w_G"] = w[1]
        named_params["w_B"] = w[2]

        return x, named_params

    def normal_weight_B(self, x):
        w = np.random.randn(3)
        x = (x @ w)[..., None]
        x = self._sigmoid(x)

        # Params
        named_params = {}
        named_params["w_R"] = w[0]
        named_params["w_G"] = w[1]
        named_params["w_B"] = w[2]

        return x, named_params

    def uniform_weight_A(self, x):
        w = np.random.uniform(0, 1, size=3)
        w = w / w.sum()
        x = (x @ w)[..., None]
        # x = self._sigmoid(x)
        # x = np.clip(x, 0, 1)

        # Params
        named_params = {}
        named_params["w_R"] = w[0]
        named_params["w_G"] = w[1]
        named_params["w_B"] = w[2]

        return x, named_params

    def shrink(self, x, coef=0.1):
        g = (x @ self.w_classic)[..., None]

        shrink = np.random.uniform(0, 1)
        bias = np.random.uniform(0, 1)

        shrink = max(shrink, 0.1) * coef
        bias = min(bias, 1.0 - shrink)
        g = g * shrink + bias

        # Params
        named_params = {}
        named_params["shrink"] = shrink
        named_params["bias"] = bias

        return g, named_params

    def scale(self, x):
        g = (x @ self.w_classic)[..., None]

        shrink = np.random.uniform(0, 1)
        bias = np.random.uniform(0, 1)

        shrink = max(shrink, 0.1)
        bias = min(bias, 1.0 - shrink)
        g = g * shrink + bias

        # Params
        named_params = {}
        named_params["shrink"] = shrink
        named_params["bias"] = bias

        return g, named_params

    def scale_with_invert(self, x):
        g = (x @ self.w_classic)[..., None]

        shrink = np.random.uniform(0, 1)
        bias = np.random.uniform(0, 1)

        shrink = max(shrink, 0.1)
        bias = min(bias, 1.0 - shrink)
        g = g * shrink + bias

        g = self._random_invert(g)

        # Params
        named_params = {}
        named_params["shrink"] = shrink
        named_params["bias"] = bias

        return g, named_params

    def classic(self, x):
        named_params = {}
        named_params["w_R"] = self.w_classic[0]
        named_params["w_G"] = self.w_classic[1]
        named_params["w_B"] = self.w_classic[2]
        return (x @ self.w_classic)[..., None], named_params

    def __call__(self, x):
        x = self.preprocess(x)
        x, named_param = self.fn_defualt(x)
        x = self.postprocess(x)
        return x

    def _random_sign(self):
        return (-1)**random.randint(0, 1)

    def _random_invert(self, x):
        if random.randint(0, 1) % 2 == 0:
            return x
        return 1 - x

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
