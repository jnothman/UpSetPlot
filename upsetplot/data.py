from __future__ import print_function, division, absolute_import

import pandas as pd
import numpy as np


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


def from_memberships(memberships, data=None):
    """Load data where each sample has a collection of set names

    The output should be suitable for passing to `UpSet` or `plot`.

    Parameters
    ----------
    memberships : sequence of collections of strings
        Each element corresponds to a data point, indicating the sets it is a
        member of.  Each set is named by a string.
    data : Series or DataFrame-like, optional
        If given, the index of set memberships is attached to this data.
        It must have the same length as `memberships`.
        If not given, the series will contain the value 1.

    Returns
    -------
    DataFrame or Series
        `data` is returned with its index indicating set membership.
        The index will have levels ordered by set names.

    Examples
    --------
    >>> from upsetplot import from_memberships
    >>> from_memberships([
    ...     ['set1', 'set3'],
    ...     ['set2', 'set3'],
    ...     ['set1'],
    ...     []
    ... ])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    set1   set2   set3
    True   False  True     1
    False  True   True     1
    True   False  False    1
    False  False  False    1
    Name: ones, dtype: ...
    >>> # now with data:
    >>> import numpy as np
    >>> from_memberships([
    ...     ['set1', 'set3'],
    ...     ['set2', 'set3'],
    ...     ['set1'],
    ...     []
    ... ], data=np.arange(12).reshape(4, 3))  # doctest: +NORMALIZE_WHITESPACE
                       0   1   2
    set1  set2  set3
    True  False True   0   1   2
    False True  True   3   4   5
    True  False False  6   7   8
    False False False  9  10  11
    """
    df = pd.DataFrame([{name: True for name in names}
                       for names in memberships])
    for set_name in df.columns:
        if not hasattr(set_name, 'lower'):
            raise ValueError('Set names should be strings')
    if df.shape[1] == 0:
        raise ValueError('Require at least one set. None were found.')
    df.sort_index(axis=1, inplace=True)
    df.fillna(False, inplace=True)
    df = df.astype(bool)
    df.set_index(list(df.columns), inplace=True)
    if data is None:
        return df.assign(ones=1)['ones']

    if hasattr(data, 'loc'):
        data = data.copy(deep=False)
    else:
        data = pd.DataFrame(data)
    if len(data) != len(df):
        raise ValueError('memberships and data must have the same length. '
                         'Got len(memberships) == %d, len(data) == %d'
                         % (len(memberships), len(data)))
    data.index = df.index
    return data
