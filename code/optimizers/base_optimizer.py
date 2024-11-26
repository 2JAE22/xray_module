from torch import optim

# Adam Optimizer
def adam_optimizer(params, **optimizer_params):
    return optim.Adam(params=params, **optimizer_params)

# SGD Optimizer
def sgd_optimizer(params, **optimizer_params):
    return optim.SGD(params=params, **optimizer_params)

# RMSprop Optimizer
def rmsprop_optimizer(params, **optimizer_params):
    return optim.RMSprop(params=params, **optimizer_params)
