from .fit import *
from .dashboard import *
from .mups_filtering import *

__version__ = '1.0'

def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
