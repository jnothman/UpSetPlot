__version__ = '0.5.dev1'

import os

if os.environ.get('__in-setup', None) != '1':
    from .plotting import UpSet, plot
    from .data import (generate_counts, generate_data, generate_samples,
                       from_memberships, from_contents)

    __all__ = ['UpSet',
               'generate_data', 'generate_counts', 'generate_samples',
               'plot',
               'from_memberships', 'from_contents']
