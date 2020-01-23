import os
import pickle
import sys

torchfuel_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(torchfuel_path)

from torchfuel.utils.state import Namespace, State


def test_ns():
    ns = Namespace()
    ns.test = 10
    print(ns.stored_objects)
    print(ns)
    plk_object = ns.pickle_safe()

    assert isinstance(plk_object, bytes)

    state = State()
    state.add_namespace("new namespace")
    assert hasattr(state, "new_namespace")


if __name__ == "__main__":
    test_ns()
