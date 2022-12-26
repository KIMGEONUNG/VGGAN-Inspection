class SignalProcessor():

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def renorm_m1_1_to_zero_1(x):
        return (x + 1) * 0.5

    @staticmethod
    def renorm_zero_1_to_m1_1(x):
        return x * 2 - 1
