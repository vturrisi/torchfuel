import pickle

from torchfuel.trainers.state import Namespace, State


def test_ns():
    ns = Namespace()
    ns.test = 10
    print(ns.stored_objects)
    print(ns)
    plk_object = ns.pickle_safe()

    assert isinstance(plk_object, bytes)

    state = State()
    state.add_namespace('new namespace')
    assert hasattr(state, 'new_namespace')


if __name__ == '__main__':
    test_ns()
