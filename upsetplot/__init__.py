__version__ = '0.3.0.post2'

import os

if os.environ.get('__in-setup', None) != '1':
    import sys
    print(os.environ, file=sys.stderr)
    from .plotting import UpSet, plot
    from .data import (generate_counts, generate_data, generate_samples,
                       from_memberships, from_contents)

    __all__ = ['UpSet',
               'generate_data', 'generate_counts', 'generate_samples',
               'plot',
               'from_memberships', 'from_contents']
