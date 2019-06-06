import glob
import os
import subprocess

import pytest


exa_glob = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'examples', '*.py')

@pytest.mark.parametrize('path', glob.glob(exa_glob))
def test_example(path):
    subprocess.check_output(['python', path])
