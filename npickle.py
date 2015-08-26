__all__ = ['read_pickle', 'write_pickle']

from pymote.logger import logger
import cPickle as pickle
import errno
import sys
import os


def _get_fh(path, mode='r'):
    """Return a file handle for given path and attempt to uncompress/compress
    files ending in '.gz'"""

    if path.endswith('.gz'):
        import gzip
        fh = gzip.open(path, mode=mode)
    else:
        fh = open(path, mode=mode)
    return fh


def write_pickle(obj, path, makedir=True):
    """Write object in Python pickle format."""
    # TODO: use normal pickling by implementing pickling protocol for Network
    # class http://docs.python.org/library/pickle.html#the-pickle-protocol
    # TODO: find out origin of maximum recursion depth problem, hack solution:
    sys.setrecursionlimit(6000)
    try:
        os.makedirs(os.path.split(path)[0])
    except OSError, e:
        if e.errno!=errno.EEXIST and e.filename!='':
            raise
    fh = _get_fh(str(path), mode='wb')
    pickle.dump(obj, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()
    logger.info('instance of %s saved in %s' % (str(obj.__class__), path))

write_npickle = write_pickle


def read_pickle(path, not_found_raises=True):
    """
    Read object in Python pickle format. If not_found_raises is True then raise
    an exception if file is missing.
    """
    try:
        fh = _get_fh(str(path), 'rb')
        obj = pickle.load(fh)
        logger.info('instance of %s loaded: %s' % (str(obj.__class__), path))
        return obj
    except IOError, e:
        # if error is some other than errno.ENOENT ='file not found raise
        if not_found_raises or e.errno!=errno.ENOENT:
            raise
        return None

read_npickle = read_pickle


# scipy.stats.norm (scipy.stats.distributions.norm_gen) object has some bounded
# (instance) methods that needs to be pickled
# this is solution for pickling instance methods found at
# http://stackoverflow.com/a/1816969/1247955
def _pickle_method(method):
    #print 'pickling',
    #print method
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
