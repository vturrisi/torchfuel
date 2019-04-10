import pickle

from torchfuel.trainers.state import Namespace


def test_ns():
    ns = Namespace()
    ns.test = 10
    print(ns.stored_objects)
    print(ns)
    plk_object = ns.pickle_safe()

    assert isinstance(plk_object, bytes)


if __name__ == '__main__':
    test_ns()
