def pymote_equal_objects(obj1, obj2):
    """
    Compare two objects and their attributes, but allow for non immutable
    attributes to be equal up to their class.
    """
    classes = obj1.__class__ == obj2.__class__
    attr_names = attr_values = True
    if isinstance(obj1, object) and isinstance(obj2, object):
        attr_names = set(obj1.__dict__.keys()) == set(obj2.__dict__.keys())
    types = (str, tuple, int, long, bool, float, frozenset, bytes, complex)
    for key, value in obj1.__dict__.items():
        other_value = getattr(obj2, key, None)
        if (isinstance(value, types) and value!=other_value) or \
            value.__class__!=other_value.__class__:
            attr_values = False
            break
    return classes and attr_names and attr_values
