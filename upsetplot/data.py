from __future__ import print_function, division, absolute_import
from numbers import Number
import functools
import distutils
import warnings
import re

import pandas as pd
import numpy as np


_concat = pd.concat
if distutils.version.LooseVersion(pd.__version__) >= '0.23.0':
    # silence the warning
    _concat = functools.partial(_concat, sort=False)


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


def _memberships_to_indicators(memberships):
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
    return df


def _contents_to_indicators(contents):
    cat_series = [pd.Series(True, index=list(elements), name=name)
                  for name, elements in contents.items()]
    if not all(s.index.is_unique for s in cat_series):
        raise ValueError('Got duplicate ids in a category')

    df = _concat(cat_series, axis=1)
    df.fillna(False, inplace=True)
    return df


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
    ... ])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
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
    ... ], data=np.arange(12).reshape(4, 3))  # doctest: +NORMALIZE_WHITESPACE
                       0   1   2
    cat1  cat2  cat3
    True  False True   0   1   2
    False True  True   3   4   5
    True  False False  6   7   8
    False False False  9  10  11
    """
    df = _memberships_to_indicators(memberships)
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
    df = _contents_to_indicators(contents)
    cat_names = list(df.columns)
    if id_column in df.columns:
        raise ValueError('A category cannot be named %r' % id_column)

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
        df = _concat([data, df], axis=1)
    df.index.name = id_column
    return df.reset_index().set_index(cat_names)


### SPEC

# TODO: Test use of CategorizedData and CategorizedCounts passed to plot()

class CategorizedData:
    """Represents data where each sample is assigned to one or more categories
    """

    def __init__(self, data, categories):
        data = pd.DataFrame(data)

        if hasattr(categories, 'dtype'):
            categories = pd.DataFrame(categories)

            invalid = set(categories.columns) & set(data.columns)
            if invalid:
                raise ValueError('Category names and data columns must be '
                                 'unique. Got overlap: %r' % sorted(invalid))

            not_in_data = categories.drop(data.index, axis=0, errors='ignore')
            if len(not_in_data):
                raise ValueError('Found identifiers in categories that are '
                                 'not in data: %r' % not_in_data.index.values)

            data = _concat([categories, data])
            categories = categories.columns

        self.data = data
        self.categories = categories
        if not categories:
            raise ValueError('Need at least one entry in categories')
        if not (set(categories) <= set(data.columns)):
            missing = sorted(set(data.columns) - set(categories))
            raise ValueError('categories should be a subset of '
                             'data.columns. '
                             'Not in data.columns: {!r}'.format(missing))
        for col in categories:
            if data[col].dtype.kind != 'b':
                raise ValueError('categories should have boolean '
                                 'dtype. Column {!r} has dtype {!r}.'.format(
                                     col, data[col].dtype.kind))

    @classmethod
    def from_memberships(cls, memberships, data=None):
        indicators = _memberships_to_indicators(memberships)
        if data is None:
            data = indicators[[]]
        return cls(data=data, categories=indicators)

    @classmethod
    def from_memberships_str(cls, memberships, data=None,
                             sep=re.compile(r'(?u)[^\w\ ]')):
        if isinstance(memberships, str):
            memberships = data[memberships]
        if hasattr(sep, 'match'):
            lists = pd.Series(memberships).apply(lambda x: sep.split)
        else:
            lists = pd.Series(memberships).str.split(sep)
        return cls.from_memberships(lists, data)

    @classmethod
    def from_contents(cls, contents, data=None):
        indicators = _contents_to_indicators(contents)
        if data is None:
            data = indicators[[]]
        return cls(data=data, categories=indicators)

    def get_counts(self, weight=None):
        gb = self.frame.groupby(self.categories)
        if weight is None:
            return CategorizedCounts(gb.size())
        else:
            return CategorizedCounts(gb[weight].sum())


class CategorizedCounts:

    def __init__(self, sizes):
        # TODO: check index is boolean and unique
        self.sizes = sizes


class OldVennData:
    def __init__(self, df, key_fields=None, category_fields=None):
        self._df = self._check_df(df)

    def _check_df(self, df):
        # TODO
        return df

    @classmethod
    def from_memberships(cls, memberships, data=None):
        """Build data from the category membership of each element

        Parameters
        ----------
        memberships : sequence of collections of strings
            Each element corresponds to a data point, indicating the sets it is
            a member of.  Each set is named by a string.
        data : Series-like or DataFrame-like, optional
            If given, the index of set memberships is attached to this data.
            It must have the same length as `memberships`.
            If not given, the series will contain the value 1.

        Returns
        -------
        VennData
        """
        return cls(from_memberships(memberships, data))

    @classmethod
    def from_contents(cls, contents, data=None):
        """Build data from category listings

        Parameters
        ----------
        contents : Mapping of strings to sets
            Map values be sets of identifiers (int or string).
        data : DataFrame, optional
            If provided, this should be indexed by the identifiers used in
            `contents`.

        Returns
        -------
        VennData
        """
        return cls(from_contents(contents, data))

    def _get_cat_mask(self):
        return self._df.index.to_frame(index=False)

    def _get_data(self):
        return self._df.reset_index()

    def get_intersection(self, categories, inclusive=False):
        """Retrieve elements that are in all the given categories

        Parameters
        ----------
        categories : collection of strings
        inclusive : bool
            If False (default), do not include elements that are in additional
            categories.
        """
        categories = list(categories)
        cat_mask = self._get_cat_mask()
        # XXX: More efficient with a groupby?
        mask = cat_mask[categories].all(axis=1)
        if not inclusive:
            mask &= ~cat_mask.drop(categories, axis=1).any(axis=1)
        return self._get_data()[mask]

    def count_intersection(self, categories, inclusive=False):
        """Count the number of elements in all the given categories

        Parameters
        ----------
        categories : collection of strings
        inclusive : bool
            If False (default), do not include elements that are in additional
            categories.
        """
