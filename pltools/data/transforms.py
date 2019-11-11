import typing
import torch


class ToTensor(object):
    def __init__(self,
                 keys: typing.Iterable = None,
                 dtypes: typing.Iterable[tuple] = None):
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, **data) -> dict:
        _keys = self._resolve_keys(data)

        torch_data = {}
        for key, item in data.items():
            if key in _keys:
                torch_data[key] = torch.from_numpy(item)
            else:
                torch_data[key] = item

        if self.dtypes is not None:
            for _dtype in self.dtypes:
                torch_data[_dtype[0]] = torch_data[_dtype[0]].to(
                    dtype=_dtype[1])

        return torch_data

    def _resolve_keys(self, data) -> typing.Iterable:
        if self.keys is None:
            _keys = data.keys()
        else:
            _keys = self.keys
        return _keys
