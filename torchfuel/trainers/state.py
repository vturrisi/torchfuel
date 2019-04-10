import pickle


class Namespace:
    def __init__(self):
        self.stored_objects = {}

    def __repr__(self):
        arg = ','.join(self.stored_objects.keys())
        return 'Namespace({})'.format(arg)

    def __setattr__(self, name, value):
        if name != 'stored_objects':
            self.stored_objects[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        '''
        intercept lookups which would raise an exception
        to check if variable is being stored
        '''
        if name in self.stored_objects:
            return self.stored_objects[name]
        else:
            return super().__getattr__(name)

    def pickle_safe(self):
        return pickle.dumps(self)


class State:
    def __init__(self):
        self.train = Namespace()
        self.eval = Namespace()
        self.test = Namespace()

    def add_namespace(self, name):
        setattr(self, name, Namespace())
