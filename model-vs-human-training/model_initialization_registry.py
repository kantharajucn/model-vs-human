from collections import defaultdict

__all__ = ['list_models', 'register']

_model__initialization_registry = []  # mapping of model names to entrypoint fns


def register(func):
    model_name = func.__name__
    _model__initialization_registry.append(model_name)

    def inner_decorator(model):
        return func(model)
    return inner_decorator


def list_models():
    """ Return list of available model names, sorted alphabetically
    """
    return _model__initialization_registry
