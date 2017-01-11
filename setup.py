# Kyuhong Shim

import sys
import lemontree
from setuptools import setup, find_packages

if sys.version_info[:2] < (3, 4):
    raise Exception('LemonTree needs Python 3.4 or later.')

setup(name='lemontree',
      version=lemontree.__version__,
      description='lemontree for deep learning',
      author='Kyuhong Shim',
      author_email='skhu20@snu.ac.kr',
      url='https://github.com/khshim/lemontree',
      license='MIT',
      packages=find_packages(),
      setup_requires=['theano'],
      install_requires=['theano'],
      )