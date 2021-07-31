from __future__ import print_function, division, absolute_import
from numbers import Number
import functools
import distutils
import warnings

import pandas as pd
import numpy as np


def generate_samples(seed=0, n_samples=10000, n_categories=3):
    """Generate artificial samples assigned to set intersections

    Parameters
    ----------
    seed : int
        A seed for randomisation
    n_samples : int
        Number of samples to generate
    n_categories : int
        Number of categories (named "cat0", "cat1", ...) to generate

    Returns
    -------
    DataFrame
        Field 'value' is a weight or score for each element.
        Field 'index' is a unique id for each element.
        Index includes a boolean indicator mask for each category.

        Note: Further fields may be added in future versions.

    See Also
    --------
    generate_counts : Generates the counts for each subset of categories
        corresponding to these samples.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({'value': np.zeros(n_samples)})
    for i in range(n_categories):
        r = rng.rand(n_samples)
        df['cat%d' % i] = r > rng.rand()
        df['value'] += r

    df.reset_index(inplace=True)
    df.set_index(['cat%d' % i for i in range(n_categories)], inplace=True)
    return df


def generate_counts(seed=0, n_samples=10000, n_categories=3):
    """Generate artificial counts corresponding to set intersections

    Parameters
    ----------
    seed : int
        A seed for randomisation
    n_samples : int
        Number of samples to generate statistics over
    n_categories : int
        Number of categories (named "cat0", "cat1", ...) to generate

    Returns
    -------
    Series
        Counts indexed by boolean indicator mask for each category.

    See Also
    --------
    generate_samples : Generates a DataFrame of samples that these counts are
        derived from.
    """
    df = generate_samples(seed=seed, n_samples=n_samples,
                          n_categories=n_categories)
    return df.value.groupby(level=list(range(n_categories))).count()


def generate_data(seed=0, n_samples=10000, n_sets=3, aggregated=False):
    warnings.warn('generate_data was replaced by generate_counts in version '
                  '0.3 and will be removed in version 0.4.',
                  DeprecationWarning)
    if aggregated:
        return generate_counts(seed=seed, n_samples=n_samples,
                               n_categories=n_sets)
    else:
        return generate_samples(seed=seed, n_samples=n_samples,
                                n_categories=n_sets)['value']


def from_indicators(indicators, data=None):
    """Load category membership indicated by a boolean indicator matrix

    This loader also supports the case where the indicator columns can be
    derived from `data`.

    .. versionadded: 0.6

    Parameters
    ----------
    indicators : DataFrame-like of booleans, Sequence of str, or callable
        Specifies the category indicators (boolean mask arrays) within
        ``data``, i.e. which records in ``data`` belong to which categories.

        If a list of strings, these should be column names found in ``data``
        whose values are boolean mask arrays.

        If a DataFrame, its columns should correspond to categories, and its
        index should be a subset of those in ``data``, values should be True
        where a data record is in that category, and False or NA otherwise.

        If callable, it will be applied to ``data`` after the latter is
        converted to a Series or DataFrame.

    data : Series-like or DataFrame-like, optional
        If given, the index of category membership is attached to this data.
        It must have the same length as `indicators`.
        If not given, the series will contain the value 1.

    Returns
    -------
    DataFrame or Series
        `data` is returned with its index indicating category membership.
        It will be a Series if `data` is a Series or 1d numeric array or None.

    Notes
    -----
    Categories with indicators that are all False will be removed.

    Examples
    --------
    >>> import pandas as pd
    >>> from upsetplot import from_indicators

    Just indicators
    >>> indicators = {"cat1": [True, False, True, False],
    ...               "cat2": [False, True, False, False],
    ...               "cat3": [True, True, False, False]}
    >>> from_indicators(indicators)
    cat1   cat2   cat3
    True   False  True     1.0
    False  True   True     1.0
    True   False  False    1.0
    False  False  False    1.0
    Name: ones, dtype: float64

    Where indicators are included within data, specifying columns by name
    >>> data = pd.DataFrame({"value": [5, 4, 6, 4], **indicators})
    >>> from_indicators(["cat1", "cat3"], data=data)
                 value   cat1   cat2   cat3
    cat1  cat3
    True  True       5   True  False   True
    False True       4  False   True   True
    True  False      6   True  False  False
    False False      4  False  False  False

    Making indicators out of all boolean columns
    >>> from_indicators(lambda data: data.select_dtypes(bool), data=data)
                       value   cat1   cat2   cat3
    cat1  cat2  cat3
    True  False True       5   True  False   True
    False True  True       4  False   True   True
    True  False False      6   True  False  False
    False False False      4  False  False  False

    Using a dataset with missing data, we can use missingness as an indicator
    >>> data = pd.DataFrame({"val1": [pd.NA, .7, pd.NA, .9],
    ...                      "val2": ["male", pd.NA, "female", "female"],
    ...                      "val3": [pd.NA, pd.NA, 23000, 78000]})
    >>> from_indicators(pd.isna, data=data)
                       val1    val2   val3
    val1  val2  val3
    True  False True   <NA>    male   <NA>
    False True  True    0.7    <NA>   <NA>
    True  False False  <NA>  female  23000
    False False False   0.9  female  78000
    """
    if data is not None:
        data = _convert_to_pandas(data)

    if callable(indicators):
        if data is None:
            raise ValueError("data must be provided when indicators is "
                             "callable")
        indicators = indicators(data)

    try:
        indicators[0]
    except Exception:
        pass
    else:
        if isinstance(indicators[0], (str, int)):
            if data is None:
                raise ValueError("data must be provided when indicators are "
                                 "specified as a list of columns")
            if isinstance(indicators, tuple):
                raise ValueError("indicators as tuple is not supported")
            # column array
            indicators = data[indicators]

    indicators = pd.DataFrame(indicators).fillna(False).infer_objects()
    # drop all-False (should we be dropping all-True also? making an option?)
    indicators = indicators.loc[:, indicators.any(axis=0)]

    if not all(dtype.kind == 'b' for dtype in indicators.dtypes):
        raise ValueError('The indicators must all be boolean')

    if data is not None:
        if not (isinstance(indicators.index, pd.RangeIndex)
                and indicators.index[0] == 0
                and indicators.index[-1] == len(data) - 1):
            # index is specified on indicators. Need to align it to data
            if not indicators.index.isin(data.index).all():
                raise ValueError("If indicators.index is not the default, "
                                 "all its values must be present in "
                                 "data.index")
            indicators = indicators.reindex(index=data.index, fill_value=False)
    else:
        data = pd.Series(np.ones(len(indicators)), name="ones")

    indicators.set_index(list(indicators.columns), inplace=True)
    data.index = indicators.index

    return data


def _convert_to_pandas(data, copy=True):
    is_series = False
    if hasattr(data, 'loc'):
        if copy:
            data = data.copy(deep=False)
        is_series = data.ndim == 1
    elif len(data):
        try:
            is_series = isinstance(data[0], Number)
        except KeyError:
            is_series = False
    if is_series:
        data = pd.Series(data)
    else:
        data = pd.DataFrame(data)
    return data


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
    ...     ['cat1', 'cat3'],
    ...     ['cat2', 'cat3'],
    ...     ['cat1'],
    ...     []
    ... ])
    cat1   cat2   cat3
    True   False  True     1
    False  True   True     1
    True   False  False    1
    False  False  False    1
    Name: ones, dtype: ...
    >>> # now with data:
    >>> import numpy as np
    >>> from_memberships([
    ...     ['cat1', 'cat3'],
    ...     ['cat2', 'cat3'],
    ...     ['cat1'],
    ...     []
    ... ], data=np.arange(12).reshape(4, 3))
                       0   1   2
    cat1  cat2  cat3
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

    data = _convert_to_pandas(data)
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
    contents : Mapping (or iterable over pairs) of strings to sets
        Keys are category names, values are sets of identifiers (int or
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

    Notes
    -----
    The order of categories in the output DataFrame is determined from
    `contents`, which may have non-deterministic iteration order.

    Examples
    --------
    >>> from upsetplot import from_contents
    >>> contents = {'cat1': ['a', 'b', 'c'],
    ...             'cat2': ['b', 'd'],
    ...             'cat3': ['e']}
    >>> from_contents(contents)
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
    >>> from_contents(contents, data=data)
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

    concat = pd.concat
    if distutils.version.LooseVersion(pd.__version__) >= '0.23.0':
        # silence the warning
        concat = functools.partial(concat, sort=False)

    df = concat(cat_series, axis=1)
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
        not_in_data = df.drop(data.index, axis=0, errors='ignore')
        if len(not_in_data):
            raise ValueError('Found identifiers in contents that are not in '
                             'data: %r' % not_in_data.index.values)
        df = df.reindex(index=data.index).fillna(False)
        df = concat([data, df], axis=1)
    df.index.name = id_column
    return df.reset_index().set_index(cat_names)
