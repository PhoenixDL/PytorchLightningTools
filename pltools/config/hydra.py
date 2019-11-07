from omegaconf import DictConfig

DICT_TYPES = (dict, DictConfig)


def nested_get(dict_like, key):
    values = []
    for _key, _item in dict_like.items():
        if key == _key:
            values.append(_item)
        elif type(_item) in DICT_TYPES:
            values.extend(nested_get(dict_like[_key], key))
    return values


def nested_contains(dict_like, key):
    val = nested_get_first(dict_like, key)
    if not isinstance(val, list) or val:
        return True
    else:
        return False


def nested_get_first(dict_like, key):
    # uses depth first search to find first key
    for _key, _item in dict_like.items():
        if key == _key:
            return _item
        elif type(_item) in DICT_TYPES:
            return nested_get_first(dict_like[_key], key)
    return []


def patch_dictconf():
    no_duplicate = True
    if not hasattr(DictConfig, "nested_get"):
        DictConfig.nested_get = nested_get
    else:
        no_duplicate = False

    if not hasattr(DictConfig, "nested_contains"):
        DictConfig.nested_contains = nested_contains
    else:
        no_duplicate = False

    if not hasattr(DictConfig, "nested_get_first"):
        DictConfig.nested_get_first = nested_get_first
    else:
        no_duplicate = False
    return no_duplicate
