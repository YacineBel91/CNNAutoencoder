import torch

class AdamOptim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, params):
        return torch.optim.Adam(params, **self.kwargs)


class ASGDOptim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, params):
        return torch.optim.ASGD(params, **self.kwargs)


class RMSPropOptim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, params):
        return torch.optim.RMSprop(params, **self.kwargs)

optimizers = {
    "Adam" : AdamOptim,
    "ASGD" : ASGDOptim,
    "RMSProp" : RMSPropOptim
}