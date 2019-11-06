import os
import typing
from pltools.data import DataContainer


def kfold_scheduler(single_run_fn: typing.Callable,
                    container_gen: DataContainer,
                    *args, **kwargs) -> typing.List[typing.Any]:
    initial_cwd = os.getcwd()
    results = []
    for _container in container_gen:
        new_dir = os.path.join(initial_cwd, f'fold{_container.fold}')
        os.mkdir(new_dir)
        os.chdir(new_dir)
        results.append(single_run_fn(_container.dset, *args, **kwargs))
    return results
