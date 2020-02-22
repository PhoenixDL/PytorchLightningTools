from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from pltools._debug_mode import get_current_debug_mode, switch_debug_mode, \
    set_debug_mode

# Flags
AUTOMATIC_DICT_PATCHING = True

if AUTOMATIC_DICT_PATCHING:
    from pltools.config import patch_dictconf
    assert patch_dictconf()
