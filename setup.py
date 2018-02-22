#!/usr/bin/env python

import os
import sys
from setuptools import setup

from upsetplot import __version__ as version


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    try:
        # See setup.cfg
        setup(name='UpSetPlot',
              version=version,
              packages=["upsetplot"],
              classifiers=[
                  'Development Status :: 3 - Alpha',
                  'Intended Audience :: Science/Research',
                  'Topic :: Scientific/Engineering :: Visualization',
                  'Programming Language :: Python :: 3'
                  'Programming Language :: Python :: 3.4'
                  'Programming Language :: Python :: 3.5'
                  'Programming Language :: Python :: 3.6'
                  'Programming Language :: Python :: 2.7'
              ],
              license="BSD",
              setup_requires=['pytest-runner'],
              tests_require=['pytest>=2.7', 'pytest-cov~=2.4'],
              # TODO: check versions
              install_requires=['pandas', 'matplotlib'])
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return


if __name__ == '__main__':
    setup_package()
