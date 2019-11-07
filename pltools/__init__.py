from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Flags
AUTOMATIC_DICT_PATCHING = True

if AUTOMATIC_DICT_PATCHING:
    from pltools.config import patch_dictconf
    patch_dictconf()
