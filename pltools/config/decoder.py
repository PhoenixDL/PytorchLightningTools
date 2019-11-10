import hydra
import copy
import typing
from functools import partial

from pltools.utils.search import breadth_first_multiple_search


class HydraDecoder:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decode_mapping = {
            "functionref": self.decode_functionref,
            "classref": self.decode_classref,
            "class": self.decode_class,
            "function": self.decode_function,
        }

    def __call__(self, config: dict) -> dict:
        return self.decode(config)

    def decode(self, config: dict) -> dict:
        for key, decode_fn in self.decode_mapping.items():
            decoding_keys, _ = breadth_first_multiple_search(config, key)
            decoding_keys = [k.rsplit('.', 1)[0] for k in decoding_keys]
            for _decode_key in decoding_keys:
                config[_decode_key] = decode_fn(config[_decode_key])
        return config

    @staticmethod
    def decode_functionref(config: dict) -> typing.Callable:
        return hydra.utils.get_method(config["functionref"])

    @staticmethod
    def decode_classref(config: dict) -> typing.Callable:
        return hydra.utils.get_class(config["classref"])

    @staticmethod
    def decode_class(config: dict) -> typing.Any:
        cf = copy.copy(config)
        args = cf.get("args", [])
        params = cf.pop("params", {})
        params.update(cf.pop("kwargs", {}))
        obj = hydra.utils.instantiate(config, *args, **params)
        return obj

    @staticmethod
    def decode_function(config: dict) -> typing.Callable:
        args = config.get("args", [])
        kwargs = config.get("params", {})
        return partial(hydra.utils.get_method(config["function"]), *args, **kwargs)