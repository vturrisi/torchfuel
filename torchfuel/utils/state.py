import pickle
from typing import Any, Dict, Optional


class Namespace:
    def __init__(self):
        self.stored_objects = {}

    def __repr__(self):
        arg = ",".join(self.stored_objects.keys())
        return f"Namespace({arg})"

    def __setattr__(self, name: str, value: int) -> None:
        if name != "stored_objects":
            self.stored_objects[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """
        intercept lookups which would raise an exception
        to check if variable is being stored
        """
        if name in self.stored_objects:
            return self.stored_objects[name]
        elif name == "stored_objects":
            return self.stored_objects
        else:
            raise AttributeError(
                f"'Namespace' object has no attribute {name} (shouldn't fall here)"
            )

    def pickle_safe(self) -> bytes:
        return pickle.dumps(self)


class State:
    # this stops mypy from complaining (trainer)
    current_epoch: int
    elapsed_time: float

    train_loss: float
    eval_loss: float
    test_loss: float

    train_acc: float
    eval_acc: float
    test_acc: float

    current_minibatch: Optional[Dict]
    current_minibatch_stats: Optional[Dict]

    def __init__(self):
        self.train = Namespace()
        self.eval = Namespace()
        self.test = Namespace()

    def add_namespace(self, name: str) -> None:
        assert not name[0].isdigit()
        name = name.replace(" ", "_")
        name = name.replace("-", "_")
        setattr(self, name, Namespace())
