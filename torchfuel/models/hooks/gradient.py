from torchfuel.utils.state import State


def store_gradients(module, grad_input, grad_out) -> None:
    print(module, type(module))
