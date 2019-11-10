from __future__ import annotations

import copy
import warnings
import yaml
import argparse
from pltools.config.hydra import patch_dictconf
from omegaconf import DictConfig


def non_string_warning(func):
    def warning_wrapper(config, key, *args, **kwargs):
        """
        Emit warning if non string keys are used
        Parameters
        ----------
        config: :class:`Config`
            decorated function receive :param:`self` as first argument
        key : immutable type
            key which is checked
        Returns
        -------
        callable
            original function with arguments
        """
        if not isinstance(key, str):
            warnings.warn("The key {} is not a string, but a {}. "
                          "This may lead to unwanted behavior!".format(
                              key, type(key)), RuntimeWarning)

        return func(config, key, *args, **kwargs)
    return warning_wrapper


class Config(dict):
    """
    Baseclass to create a config which hold arbitrary data
    """

    def __init__(self, dict_like=None, **kwargs):
        """
        Parameters
        ----------
        dict_like : dict, optional
            dict like object to initialize config, by default None
        kwargs:
            additional arguments added to the config
        Warnings
        --------
        It is recommended to only use strings as keys inside the config.
        Because of the shortened access to nested keys the types of the
        keys are lost.
        Examples
        --------
        Create simple configuration with nested keys
        >>> from pltools.config import Config
        >>> cf = Config()
        >>> # automatically generates new nested dictionaries
        >>> cf['first_level.second_level.third_level'] = 1
        >>> # form access
        >>> print(cf['first_level.second_level.third_level'])
        >>> # traditional access
        >>> print(cf['first_level']['second_level']['third_level'])
        >>> # entries can also be accessed with dot operator
        >>> print(cf.first_level.second_level.thirs_level)
        """

        super().__init__()
        self.__dict__ = self
        if dict_like is not None:
            self.update(dict_like)
        self.update(kwargs)
        patch_dictconf(type(self))

    @non_string_warning
    def __setattr__(self, key, value):
        """
        Set attribute in config
        Parameters
        ----------
        key : str
            attribute name
        value : any
            attribute value
        """
        super().__setattr__(key, self._to_config(value))

    @non_string_warning
    def __setitem__(self, key, value):
        """
        Set items inside dict. Supports setting of nested entries by
        seperating the individual keys with a '.'.
        Parameters
        ----------
        key : str
            key for new value
        value : any
            new value
        """
        if not isinstance(key, str) or '.' not in key:
            super().__setitem__(key, value)
        else:
            current_level = self
            keys = key.split(".")
            final_key = keys.pop(-1)
            final_dict = self._traverse_keys(keys, create=True)
            final_dict._set_internal_item(final_key, value)

    def _traverse_keys(self, keys, create=False):
        """
        Internal helper to traverse through nested dicts
        (iterative implementation to avoid problems with python stack)
        Parameters
        ----------
        keys : iterable of list
            iterable with keys which should be traversed
        create : bool, optional
            creates new empty configs for non existant keys, by default False
        Returns
        -------
        Any
            value defined by the traversed keys
        """
        current_level = self
        for k in keys:
            if k not in current_level:
                if create:
                    current_level[k] = self._create_internal_dict()
                else:
                    raise KeyError(
                        "{} was not found in internal dict.".format(k))
            # traverse to needed dict
            current_level = current_level[k]
        return current_level

    def _set_internal_item(self, key, item, deepcopy=False):
        """
        Set internal item
        Parameters
        ----------
        key : str
            key where new item should be assigned
        item : Any
            item which should be assigned
        deepcopy : bool, optional
            if enabled the item is copied to the config, by default False
        """
        config_item = self._to_config(item)
        if deepcopy:
            self[key] = copy.deepcopy(config_item)
        else:
            self[key] = config_item

    @classmethod
    def _to_config(cls, item):
        """
        Convert items to config if they are a dict like object
        but not already a config
        Parameters
        ----------
        item : Any
            item which is converted
        Returns
        -------
        Any
            return a config is item is dict like, otherwise the item is
            returned
        """
        if isinstance(item, dict) and not isinstance(item, cls):
            # convert dict to config for additional functionality
            return cls._create_internal_dict(item)
        else:
            return item

    @staticmethod
    def _create_internal_dict(*args, **kwargs):
        """
        Defines how internal dicts should be created. Can be used to easily
        overwrite subclasses
        Returns
        -------
        :class:`Config`
            new config
        """
        return Config(*args, **kwargs)

    @non_string_warning
    def __getitem__(self, key):
        """
        Get single item
        Parameters
        ----------
        key : str
            key to desired item
        Returns
        -------
        Any
            value inside dict
        """
        if not isinstance(key, str) or '.' not in key:
            try:
                return super().__getitem__(int(key))
            except (KeyError, ValueError):
                return super().__getitem__(key)
        else:
            return self._traverse_keys(key.split("."), create=False)

    @non_string_warning
    def __contains__(self, key):
        """
        Check if key is in config
        (also works for nested dicts with short form)
        Parameters
        ----------
        key : str
            key for desired value
        Returns
        -------
        bool
            true if key is in config
        """
        contain = True
        try:
            self[key]
        except KeyError:
            contain = False
        return contain

    def update(self, update_dict, deepcopy=False, overwrite=False):
        """
        Update internal dicts with dict like object
        Parameters
        ----------
        update_dict : dictlike
            values which should be added to config
        deepcopy : bool, optional
            copies values from :param:`update_dict`, by default False
        overwrite : bool, optional
            overwrite existing values inside config, by default False
        Raises
        ------
        ValueError
            if overwrite is not enabled and `update_dict` contains same values
            as config
        """
        for key, item in update_dict.items():
            # update items individually
            self._update(key, item, deepcopy=deepcopy, overwrite=overwrite)

    def _update(self, key, item, deepcopy=False, overwrite=False):
        """
        Helper function for update
        Parameters
        ----------
        key : str
            key where new item should be assigned
        item : Any
            item which should be assigned
        deepcopy : bool, optional
            copies :param:`item`, by default False
        overwrite : bool, optional
            overwrite existing values inside config, by default False
        """
        if isinstance(item, dict) or isinstance(item, DictConfig):
            # update nested dicts
            if key not in self:
                self[key] = self._create_internal_dict({})
            self[key].update(item, deepcopy=deepcopy, overwrite=overwrite)
        else:
            # check for overwrite
            self._raise_overwrite(key, overwrite=overwrite)
            # set item
            self._set_internal_item(key, item, deepcopy=deepcopy)

    def _raise_overwrite(self, key, overwrite):
        """
        Checks if a ValueError should be raised
        Parameters
        ----------
        key : str
            key which needs to be checked
        overwrite : bool
            if overwrite is enabled no ValueError is raised even if the key
            already exists
        Raises
        ------
        ValueError
            raised if overwrite is not enabled and key already exists
        """
        if key in self and not overwrite:
            raise ValueError("{} already in config. Can "
                             "not overwrite value.".format(key))

    # def dump(self, path, formatter=yaml.dump, **kwargs):
    #     """
    #     Save config to a file and add time stamp to config
    #     Parameters
    #     ----------
    #     path : str
    #         path where config is saved
    #     formatter : callable, optional
    #         defines the format how the config is saved, by default yaml.dump
    #     kwargs:
    #         additional keyword arguments passed to :param:`formatter`
    #     """
    #     with open(path, "w") as f:
    #         formatter(self, f, **kwargs)
    #
    # def dumps(self, formatter=yaml.dump, **kwargs):
    #     """
    #     Create a loadable string representation from the config and
    #     add time stamp to config
    #     Parameters
    #     ----------
    #     formatter : callable, optional
    #         defines the format how the config is saved, by default yaml.dump
    #     kwargs:
    #         additional keyword arguments passed to :param:`formatter`
    #     """
    #     return formatter(self, **kwargs)

    def load(self, path, formatter=yaml.load, **kwargs):
        """
        Update config from a file
        Parameters
        ----------
        path : str
            path to file
        formatter : callable, optional
            defines the format how the config is saved, by default yaml.dump
        kwargs:
            additional keyword arguments passed to :param:`formatter`
        """
        with open(path, "r") as f:
            updater = formatter(f, **kwargs)
        self.update(updater, overwrite=True)

    def loads(self, data, formatter=yaml.load, **kwargs):
        """
        Update config from a string
        Parameters
        ----------
        data: str
            string representation of config
        formatter : callable, optional
            defines the format how the config is saved, by default yaml.dump
        kwargs:
            additional keyword arguments passed to :param:`formatter`
        """
        updater = formatter(data, **kwargs)
        self.update(updater, overwrite=True)

    @classmethod
    def create_from_dict(cls, value, deepcopy=False):
        """
        Create config from dict like object
        Parameters
        ----------
        value : dict like
            dict like object used to create new config
        deepcopy : bool, optional
            if enabled, copies values from origin, by default False
        Returns
        -------
        :class:`Config`
            new config
        Raises
        ------
        TypeError
            raised if :param:`value` is not a dict (or a subclass of dict)
        """
        if not isinstance(value, dict):
            raise TypeError("Value must be an instance of dict but type {} "
                            "was found.".format(type(value)))
        config = cls()
        config.update(value, deepcopy=deepcopy)
        return config

    @classmethod
    def create_from_argparse(cls, value, deepcopy=False, **kwargs):
        """
        Create config from argument parser
        Parameters
        ----------
        value : argument parser or namespace
            if value is an argument parser, the arguments are first parsed
            and than a new config with the values is created
            if value is a Namespace the new config is created immediatly
        deepcopy : bool, optional
            if enabled, copies values from origin, by default False
        Returns
        -------
        :class:`Config`
            new config
        Raises
        ------
        TypeError
            if value is not an instance of :class:`ArgumentParser`
            or :class:`Namespace`
        """
        if isinstance(value, argparse.ArgumentParser):
            args_parsed = value.parse_args(**kwargs)
            return cls.create_from_argparse(args_parsed, deepcopy=deepcopy)
        elif isinstance(value, argparse.Namespace):
            return cls.create_from_dict(vars(value), deepcopy=deepcopy)
        else:
            raise TypeError("Type of args not supported.")

    @classmethod
    def create_from_file(cls, path, formatter=yaml.load, **kwargs):
        """
        Create config from a file
        Parameters
        ----------
        path : str
            path to file
        formatter : callable, optional
            defines the format how the config is saved, by default yaml.dump
        kwargs:
            additional keyword arguments passed to :param:`formatter`
        Returns
        -------
        :class:`Config`
            new config
        """
        config = cls()
        config.load(path, formatter=formatter, **kwargs)
        return config

    @classmethod
    def create_from_hydra(cls, cfg: DictConfig, deepcopy: bool = False) -> Config:
        config = cls()
        config.update(cfg, deepcopy=deepcopy)
        return config

    @classmethod
    def create_from_str(cls, data, formatter=yaml.load, **kwargs):
        """
        Create config from a string
        Parameters
        ----------
        data: str
            string representation of config
        formatter : callable, optional
            defines the format how the config is saved, by default yaml.dump
        kwargs:
            additional keyword arguments passed to :param:`formatter`
        Returns
        -------
        :class:`Config`
            new config
        """
        config = cls()
        config.loads(data, formatter=formatter, **kwargs)
        return config
