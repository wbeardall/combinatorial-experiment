from dill import Unpickler

class SafeUnpickler(Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module,name)
        except ModuleNotFoundError as e:
            if name == "experiment_function":
                return type(None)
            else:
                raise e
            
def safe_load(file, ignore=None, **kwds):
    """
    Unpickle an object from a file.

    See :func:`loads` for keyword arguments.
    """
    return SafeUnpickler(file, ignore=ignore, **kwds).load()