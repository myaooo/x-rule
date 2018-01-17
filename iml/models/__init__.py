import pickle

FILE_EXTENSION = 'mdl'


def format_name(name):
    return f"{name}.{FILE_EXTENSION}"


class ModelBase:
    def __init__(self, name):
        self.name = name

    @property
    def type(self):
        raise NotImplementedError("Base class")

    def train(self, x, y):
        raise NotImplementedError("Base class")

    def test(self, x, y):
        raise NotImplementedError("Base class")

    def save(self, filename=None):
        if filename is None:
            filename = format_name(self.name)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def infer(self, x):
        raise NotImplementedError("Base class")

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            mdl = pickle.load(f)
            if isinstance(mdl, cls):
                return mdl
            else:
                raise RuntimeError("The loaded file is not a Tree model!")
