import typing


def breadth_first_multiple_search(
        dict_like: dict, key: str, prev_key: str = '', delimiter: str = '.') -> (
            typing.List[str], typing.List[typing.Any]):
    found_keys = []
    found_items = []
    for _key, _item in dict_like.items():
        new_key = prev_key + delimiter + str(_key) if prev_key else prev_key + str(_key)
        if _key == key:
            found_keys.append(new_key)
            found_items.append(_item)
        else:
            if isinstance(_item, dict):
                _subkeys, _subitems = breadth_first_multiple_search(
                    dict_like[_key], key, new_key)
                found_keys.extend(_subkeys)
                found_items.extend(_subitems)

    return found_keys, found_items
