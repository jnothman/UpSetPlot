from __future__ import print_function, division, absolute_import

import operator
from collections import OrderedDict

import numpy as np
import pandas as pd


def generate_data(seed=0, n_samples=10000, n_sets=3, aggregated=False):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({'value': np.zeros(n_samples)})
    for i in range(n_sets):
        r = rng.rand(n_samples)
        df['set%d' % i] = r > rng.rand()
        df['value'] += r

    df.set_index(['set%d' % i for i in range(n_sets)], inplace=True)
    if aggregated:
        return df.value.groupby(level=list(range(n_sets))).count()
    return df.value


def from_sets(data):
    """
    Data loader for a dict of sets

    :param Dict[str, set[str]] data:
    :return: Multi-Index Series of intersection counts
    :rtype: pd.Series
    """
    # Convert dict into OrderedDict to preserve Key/Value order
    data = OrderedDict(data)

    # Construct Index
    tf_array = [[True, False]] * len(data)
    index = pd.MultiIndex.from_product(tf_array, names=data.keys())

    # Curate values from each intersection group
    values = []
    for i in index:
        values.append(_intersection_counts(data.values(), i))

    return pd.Series(values, index=index)


def _intersection_counts(sets, bool_tuple):
    """
    Given list of sets and boolean tuple, return count of intersection

    :param List[sets[str]] sets:
    :param Tuple[bool] bool_tuple:
    :return: Count of intersection
    :rtype: int
    """
    # For all False case, return 0
    if True not in bool_tuple:
        return 0

    # Operator dictionary
    set_ops = {True: operator.and_, False: operator.sub}

    # For each grouping, perform set operation
    zipped = sorted(list(zip(bool_tuple, sets)), reverse=True)
    _, base = zipped[0]
    for operation, s in zipped[1:]:
        base = set_ops[operation](base, s)

    return len(base)
