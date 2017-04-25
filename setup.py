
from setuptools import setup

from xijafit import __version__

setup(
    name='xijafit',
    version=__version__,
    author='Matthew Dahmer',
    author_email='matthew.dahmer@gmail.com',
    packages=['xijafit'],
    url='http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/xijafit/',
    license='BSD',
    description='Tool for automating the Xija model fitting process',
    long_description=open('README').read(),
    package_data={'xijafit': ['example_data/*.json']},
)