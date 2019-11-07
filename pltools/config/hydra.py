from omegaconf import DictConfig

DICT_TYPES = (dict, DictConfig)


def nested_get(dict_like, key):
    return nested_get_fn(dict_like, lambda x: x == key)


def nested_get_fn(dict_like, fn):
    values = []
    for _key, _item in dict_like.items():
        if fn(_key):
            values.append(_item)
        elif type(_item) in DICT_TYPES:
            values.extend(nested_get_fn(dict_like[_key], fn))
    return values


def nested_get_key(dict_like, item):
    # returns position in form of a dotted string
    return nested_get_key_fn(dict_like, lambda x: item == x)


def nested_get_key_fn(dict_like, fn, sep='.'):
    # returns position in form of a dotted string
    def _nested_get_key_fn(_dict_like, prefix):
        keys = []
        for _key, _item in _dict_like.items():
            if fn(_item):
                keys.append(f'{prefix}{sep}{_key}')
            elif type(_item) in DICT_TYPES:
                keys.extend(_nested_get_key_fn(_dict_like[_key],
                                               f'{prefix}{sep}{_key}'))
        return keys

    found_keys = _nested_get_key_fn(dict_like, '')
    found_keys = [k[1:] for k in found_keys]
    return found_keys


def nested_set(dict_like, key, item, create=False, sep='.'):
    # key specifies position inside dict with dots
    keys = str(key).split(sep, 1)
    if len(keys) == 1:
        dict_like[keys[0]] = item
    else:
        if keys[0] in dict_like:
            nested_set(dict_like[keys[0]], keys[1], item, create, sep)
        elif create:
            dict_like[keys[0]] = {}
            nested_set(dict_like[keys[0]], keys[1], item, create, sep)
        else:
            raise ValueError(f'{key} is not in dict-like')


def nested_contains(dict_like, key):
    for _key, _item in dict_like.items():
        if key == _key:
            return True
        elif type(_item) in DICT_TYPES:
            return nested_contains(dict_like[_key], key)
    return False


def nested_get_first(dict_like, key, default=None):
    # uses depth first search to find first key
    for _key, _item in dict_like.items():
        if key == _key:
            return _item
        elif type(_item) in DICT_TYPES:
            return nested_get_first(dict_like[_key], key)
    return default


def patch_dictconf():
    def patch_fn(fn_name, fn):
        if not hasattr(DictConfig, fn_name):
            setattr(DictConfig, fn_name, fn)
            return True
        else:
            return False

    patch_fns = (
        ("nested_get", nested_get),
        ("nested_get_fn", nested_get_fn),
        ("nested_get_key", nested_get_key),
        ("nested_get_key_fn", nested_get_key_fn),
        ("nested_set", nested_set),
        ("nested_contains", nested_contains),
        ("nested_get_first", nested_get_first)
    )

    no_duplicate = True
    for patch in patch_fns:
        no_duplicate = no_duplicate and patch_fn(patch[0], patch[1])
    return no_duplicate
