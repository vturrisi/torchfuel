import pickle

from torchfuel.trainers.state import Placeholder


def test_ph():
    ph = Placeholder()
    ph.test = 10
    print(ph.stored_objects)
    print(ph)
    plk_object = ph.pickle_safe()

    assert isinstance(plk_object, bytes)
