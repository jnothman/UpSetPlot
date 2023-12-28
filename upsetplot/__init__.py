__version__ = '0.8.2'

import os

if os.environ.get('__in-setup', None) != '1':
    from .plotting import UpSet, plot
    from .data import (generate_counts, generate_data, generate_samples,
                       from_memberships, from_contents, from_indicators)
    from .reformat import query

    __all__ = ['UpSet',
               'generate_data', 'generate_counts', 'generate_samples',
               'plot',
               'from_memberships', 'from_contents', 'from_indicators', 'query']
