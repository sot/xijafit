
from setuptools import setup

from xijafit import __version__

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(
    name='xijafit',
    version=__version__,
    author='Matthew Dahmer',
    author_email='matthew.dahmer@gmail.com',
    packages=['xijafit', 'xijafit.tests'],
    url='http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/xijafit/',
    license='BSD',
    description='Tool for automating the Xija model fitting process',
    package_data={'xijafit.tests': ['*.json']},
    tests_require=['pytest'],

)