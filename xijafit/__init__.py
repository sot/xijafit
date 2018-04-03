from .fit import *

__version__ = '0.4'

def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
