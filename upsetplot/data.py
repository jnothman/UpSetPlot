from __future__ import print_function, division, absolute_import
from numbers import Number

import pandas as pd
import numpy as np


def generate_data(seed=0, n_samples=10000, n_sets=3, aggregated=False):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({'value': np.zeros(n_samples)})
    for i in range(n_sets):
        r = rng.rand(n_samples)
        df['cat%d' % i] = r > rng.rand()
        df['value'] += r

    df.set_index(['cat%d' % i for i in range(n_sets)], inplace=True)
    if aggregated:
        return df.value.groupby(level=list(range(n_sets))).count()
    return df.value


def from_memberships(memberships, data=None):
    """Load data where each sample has a collection of category names

    The output should be suitable for passing to `UpSet` or `plot`.

    Parameters
    ----------
    memberships : sequence of collections of strings
        Each element corresponds to a data point, indicating the sets it is a
        member of.  Each category is named by a string.
    data : Series-like or DataFrame-like, optional
        If given, the index of category memberships is attached to this data.
        It must have the same length as `memberships`.
        If not given, the series will contain the value 1.

    Returns
    -------
    DataFrame or Series
        `data` is returned with its index indicating category membership.
        It will be a Series if `data` is a Series or 1d numeric array.
        The index will have levels ordered by category names.

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
            raise ValueError('Category names should be strings')
    if df.shape[1] == 0:
        raise ValueError('Require at least one category. None were found.')
    df.sort_index(axis=1, inplace=True)
    df.fillna(False, inplace=True)
    df = df.astype(bool)
    df.set_index(list(df.columns), inplace=True)
    if data is None:
        return df.assign(ones=1)['ones']

    if hasattr(data, 'loc'):
        data = data.copy(deep=False)
    elif len(data) and isinstance(data[0], Number):
        data = pd.Series(data)
    else:
        data = pd.DataFrame(data)
    if len(data) != len(df):
        raise ValueError('memberships and data must have the same length. '
                         'Got len(memberships) == %d, len(data) == %d'
                         % (len(memberships), len(data)))
    data.index = df.index
    return data


def from_contents(contents, data=None, id_column='id'):
    """Build data from category listings

    Parameters
    ----------
    contents : Mapping of strings to sets
        Keys are cateogry names, values are sets of identifiers (int or
        string).
    data : DataFrame, optional
        If provided, this should be indexed by the identifiers used in
        `contents`.
    id_column : str, default='id'
        The column name to use for the identifiers in the output.

    Returns
    -------
    DataFrame
        `data` is returned with its index indicating category membership,
        including a column named according to id_column.
        If data is not given, the order of rows is not assured.

    Examples
    --------
    >>> from upsetplot import from_contents
    >>> contents = {'cat1': ['a', 'b', 'c'],
    ...             'cat2': ['b', 'd'],
    ...             'cat3': ['e']}
    >>> from_contents(contents)  # doctest: +NORMALIZE_WHITESPACE
                      id
    cat1  cat2  cat3
    True  False False  a
          True  False  b
          False False  c
    False True  False  d
          False True   e
    >>> import pandas as pd
    >>> contents = {'cat1': [0, 1, 2],
    ...             'cat2': [1, 3],
    ...             'cat3': [4]}
    >>> data = pd.DataFrame({'favourite': ['green', 'red', 'red',
    ...                                    'yellow', 'blue']})
    >>> from_contents(contents, data=data)  # doctest: +NORMALIZE_WHITESPACE
                       id favourite
    cat1  cat2  cat3
    True  False False   0     green
          True  False   1       red
          False False   2       red
    False True  False   3    yellow
          False True    4      blue
    """
    cat_series = [pd.Series(True, index=list(elements), name=name)
                  for name, elements in contents.items()]
    if not all(s.index.is_unique for s in cat_series):
        raise ValueError('Got duplicate ids in a category')
    df = pd.concat(cat_series, axis=1, sort=False)
    if id_column in df.columns:
        raise ValueError('A category cannot be named %r' % id_column)
    df.fillna(False, inplace=True)
    cat_names = list(df.columns)

    if data is not None:
        if set(df.columns).intersection(data.columns):
            raise ValueError('Data columns overlap with category names')
        if id_column in data.columns:
            raise ValueError('data cannot contain a column named %r' %
                             id_column)
        not_in_data = df.drop(index=data.index, errors='ignore')
        if len(not_in_data):
            raise ValueError('Found identifiers in contents that are not in '
                             'data: %r' % not_in_data.index.values)
        df = df.reindex(index=data.index).fillna(False)
        df = pd.concat([data, df], axis=1, sort=False)
    df.index.name = id_column
    return df.reset_index().set_index(cat_names)
