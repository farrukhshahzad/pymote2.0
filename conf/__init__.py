"""
Values are taken first from pymote.conf.global_settings as defaults.
To override default values
- specify PYMOTE_SETTINGS_MODULE environment variable as module name to be used

or if settings are imported but not configured (accessed):
- use settings.configure(SETTING1=value1, SETTING2=value2)

or if they are configured:
- use settings.load('package.settings_module')

"""

import os
import sys
import re

from pymote.conf import global_settings
from warnings import warn
from pymote.logger import logger

ENVIRONMENT_VARIABLE = "PYMOTE_SETTINGS_MODULE"


class LazySettings(object):

    """
    A lazy proxy for either global pymote settings or custom settings object.

    The user can manually configure settings prior to using them. Otherwise,
    pymote uses the settings module pointed to by PYMOTE_SETTINGS_MODULE.

    """

    def __init__(self):
        self._wrapped = None

    def __getattr__(self, name):
        if self._wrapped is None:
            self._setup()
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        if name == "_wrapped":
            # Assign to __dict__ to avoid infinite __setattr__ loops.
            self.__dict__["_wrapped"] = value
        else:
            if self._wrapped is None:
                self._setup()
                setattr(self._wrapped, name, value)
            else:
                logger.error('For manual settings override use'
                             ' settings.configure(SETTING1=value1,'
                             ' SETTING2=value2) or PYMOTE_SETTINGS_MODULE.')

    def __delattr__(self, name):
        if name == "_wrapped":
            raise TypeError("can't delete _wrapped.")
        if self._wrapped is None:
            self._setup()
            delattr(self._wrapped, name)
        else:
            logger.error('For manual settings override use '
                         'settings.configure(SETTING1=val1, SETTING2=val2) '
                         'or PYMOTE_SETTINGS_MODULE.')

    # introspection support:
    __members__ = property(lambda self: self.__dir__())

    def __dir__(self):
        if self._wrapped is None:
            self._setup()
        return  dir(self._wrapped)

    def _setup(self, settings_module=None):
        """
        Load the settings module pointed to by the environment variable.

        This is used the first time we need any settings at all, if the user
        has not previously configured the settings manually.

        """
        if settings_module is None:
            try:
                settings_module = os.environ[ENVIRONMENT_VARIABLE]
                if not settings_module:  # If it's set but is an empty string.
                    raise KeyError
            except KeyError:
                settings_module = None
                logger.warning("Environment variable %s is undefined, using "
                               "global_settings." % ENVIRONMENT_VARIABLE)
            else:
                logger.info("Settings module is specified in environment"
                            "variable %s." % (ENVIRONMENT_VARIABLE))

        if settings_module is not None:
            logger.info("Using module %s to override global_settings."\
                         % (settings_module))

        self._wrapped = Settings(settings_module)

    def configure(self, default_settings=global_settings, **options):
        """
        Called to manually configure the settings.

        The `default_settings` parameter sets where to retrieve any unspecified
        values from (its argument must support attribute access (__getattr__)).

        """
        if self._wrapped != None:
            raise RuntimeError('Settings already configured or accessed no'
                               ' further configuration allowed.')
        holder = UserSettingsHolder(default_settings)
        for name, value in options.items():
            setattr(holder, name, value)
        self._wrapped = holder

    def load(self, module):
        """
        Load module and override settings with values in it. It works  even if
        settings have been configured already.
        """
        self._setup(module)

    def configured(self):
        """Returns True if the settings have already been configured."""
        return bool(self._wrapped)

    configured = property(configured)


class Settings(object):

    def __init__(self, settings_module=None):
        # update this dict from global settings, but only for ALL_CAPS settings
        for setting in dir(global_settings):
            if setting == setting.upper():
                logger.info('Setting %s on global value: %s' % \
                            (setting, str(getattr(global_settings, setting))))
                setattr(self, setting, getattr(global_settings, setting))

        # store the settings module in case someone later cares
        self.SETTINGS_MODULE = settings_module

        if (self.SETTINGS_MODULE):
            try:
                mod = import_module(self.SETTINGS_MODULE)
            except ImportError, e:
                raise ImportError("Could not import settings '%s' (Is it on "
                                  "sys.path? Does it have syntax errors?): %s"\
                                   % (self.SETTINGS_MODULE, e))

            for setting in dir(mod):
                if setting == setting.upper():
                    logger.info('Override %s on value in module: %s'
                                % (setting, str(getattr(mod, setting))))
                    setattr(self, setting, getattr(mod, setting))


class UserSettingsHolder(object):
    """Holder for user configured settings."""
    # SETTINGS_MODULE doesn't make much sense in the manually configured
    # (standalone) case.
    SETTINGS_MODULE = None

    def __init__(self, default_settings):
        """
        Requests for configuration variables not in this class are satisfied
        from the module specified in default_settings (if possible).
        """
        self.default_settings = default_settings

    def __getattr__(self, name):
        return getattr(self.default_settings, name)

    def __dir__(self):
        return self.__dict__.keys() + dir(self.default_settings)

    # For Python < 2.6:
    __members__ = property(lambda self: self.__dir__())

settings = LazySettings()


def _resolve_name(name, package, level):
    """Return the absolute name of the module to be imported."""
    if not hasattr(package, 'rindex'):
        raise ValueError("'package' not set to a string")
    dot = len(package)
    for x in xrange(level, 1, -1):  # @UnusedVariable
        try:
            dot = package.rindex('.', 0, dot)
        except ValueError:
            raise ValueError("attempted relative import beyond top-level "
                              "package")
    return "%s.%s" % (package[:dot], name)


def import_module(name, package=None):
    """
    Import a module.

    The `package` argument is required when performing a relative import. It
    specifies the package to use as the anchor point from which to resolve the
    relative import to an absolute import.

    """
    if name.startswith('.'):
        if not package:
            raise TypeError("Relative imports require the 'package' argument.")
        level = 0
        for character in name:
            if character != '.':
                break
            level += 1
        name = _resolve_name(name[level:], package, level)
    __import__(name)
    # if needed reload module
    try:
        module = sys.modules[name]
        # clear namespace first from old cruft
        module.__dict__.clear()
        module.__dict__['__name__'] = name
    except KeyError:
        pass
    else:
        module = reload(module)
    return sys.modules[name]
