import pickle
from typing import Dict, List, Optional, Union


class Namespace:
    def __init__(self):
        self.stored_objects = {}

    def __repr__(self):
        arg = ','.join(self.stored_objects.keys())
        return 'Namespace({})'.format(arg)

    def __setattr__(self, name: str, value: int) -> None:
        if name != 'stored_objects':
            self.stored_objects[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> List[Dict[str, Union[int, float]]]:
        '''
        intercept lookups which would raise an exception
        to check if variable is being stored
        '''
        if name in self.stored_objects:
            return self.stored_objects[name]
        else:
            return super().__getattr__(name)

    def pickle_safe(self) -> bytes:
        return pickle.dumps(self)


class State:
    # this stops mypy from complaining
    current_epoch: int
    elapsed_time: float

    train_loss: float
    eval_loss: float
    test_loss: float

    current_minibatch: Optional[dict]
    current_minibatch_stats: Optional[dict]

    def __init__(self):
        self.train = Namespace()
        self.eval = Namespace()
        self.test = Namespace()

    def add_namespace(self, name: str) -> None:
        assert not name[0].isdigit()
        name = name.replace(' ', '_')
        name = name.replace('-', '_')
        setattr(self, name, Namespace())
