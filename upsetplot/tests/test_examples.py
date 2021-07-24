import glob
import os
import subprocess
import sys

import pytest


exa_glob = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'examples', '*.py')


@pytest.mark.parametrize('path', glob.glob(exa_glob))
def test_example(path):
    pytest.importorskip('sklearn')
    pytest.importorskip('seaborn')
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    subprocess.check_output([sys.executable, path], env=env)
