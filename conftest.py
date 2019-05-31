import sys
import pytest
from _pytest.doctest import DoctestItem


def pytest_runtest_setup(item):
    if isinstance(item, DoctestItem):
        if sys.version_info.major < 3 or (sys.version_info.major == 3
                                          and sys.version_info.minor < 6):
            pytest.skip('Doctests are disabled in Python < 3.6')
