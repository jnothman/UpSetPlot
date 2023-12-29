from __future__ import print_function, division, absolute_import

try:
    import typing
except ImportError:
    import collections as typing

import numpy as np
import pandas as pd


def _aggregate_data(df, subset_size, sum_over):
    """
    Returns
    -------
    df : DataFrame
        full data frame
    aggregated : Series
        aggregates
    """
    _SUBSET_SIZE_VALUES = ['auto', 'count', 'sum']
    if subset_size not in _SUBSET_SIZE_VALUES:
        raise ValueError('subset_size should be one of %s. Got %r'
                         % (_SUBSET_SIZE_VALUES, subset_size))
    if df.ndim == 1:
        # Series
        input_name = df.name
        df = pd.DataFrame({'_value': df})

        if subset_size == 'auto' and not df.index.is_unique:
            raise ValueError('subset_size="auto" cannot be used for a '
                             'Series with non-unique groups.')
        if sum_over is not None:
            raise ValueError('sum_over is not applicable when the input is a '
                             'Series')
        if subset_size == 'count':
            sum_over = False
        else:
            sum_over = '_value'
    else:
        # DataFrame
        if sum_over is False:
            raise ValueError('Unsupported value for sum_over: False')
        elif subset_size == 'auto' and sum_over is None:
            sum_over = False
        elif subset_size == 'count':
            if sum_over is not None:
                raise ValueError('sum_over cannot be set if subset_size=%r' %
                                 subset_size)
            sum_over = False
        elif subset_size == 'sum':
            if sum_over is None:
                raise ValueError('sum_over should be a field name if '
                                 'subset_size="sum" and a DataFrame is '
                                 'provided.')

    gb = df.groupby(level=list(range(df.index.nlevels)), sort=False)
    if sum_over is False:
        aggregated = gb.size()
        aggregated.name = 'size'
    elif hasattr(sum_over, 'lower'):
        aggregated = gb[sum_over].sum()
    else:
        raise ValueError('Unsupported value for sum_over: %r' % sum_over)

    if aggregated.name == '_value':
        aggregated.name = input_name

    return df, aggregated


def _check_index(df):
    # check all indices are boolean
    if not all(set([True, False]) >= set(level)
               for level in df.index.levels):
        raise ValueError('The DataFrame has values in its index that are not '
                         'boolean')
    df = df.copy(deep=False)
    # XXX: this may break if input is not MultiIndex
    kw = {'levels': [x.astype(bool) for x in df.index.levels],
          'names': df.index.names,
          }
    if hasattr(df.index, 'codes'):
        # compat for pandas <= 0.20
        kw['codes'] = df.index.codes
    else:
        kw['labels'] = df.index.labels
    df.index = pd.MultiIndex(**kw)
    return df


def _scalar_to_list(val):
    if not isinstance(val, (typing.Sequence, set)) or isinstance(val, str):
        val = [val]
    return val


def _get_subset_mask(agg, min_subset_size, max_subset_size,
                     min_degree, max_degree,
                     present, absent):
    """Get a mask over subsets based on size, degree or category presence"""
    subset_mask = True
    if min_subset_size is not None:
        subset_mask = np.logical_and(subset_mask, agg >= min_subset_size)
    if max_subset_size is not None:
        subset_mask = np.logical_and(subset_mask, agg <= max_subset_size)
    if (min_degree is not None and min_degree >= 0) or max_degree is not None:
        degree = agg.index.to_frame().sum(axis=1)
        if min_degree is not None:
            subset_mask = np.logical_and(subset_mask, degree >= min_degree)
        if max_degree is not None:
            subset_mask = np.logical_and(subset_mask, degree <= max_degree)
    if present is not None:
        for col in _scalar_to_list(present):
            subset_mask = np.logical_and(
                subset_mask,
                agg.index.get_level_values(col).values)
    if absent is not None:
        for col in _scalar_to_list(absent):
            exclude_mask = np.logical_not(
                agg.index.get_level_values(col).values)
            subset_mask = np.logical_and(subset_mask, exclude_mask)
    return subset_mask


def _filter_subsets(df, agg,
                    min_subset_size, max_subset_size,
                    min_degree, max_degree,
                    present, absent):
    subset_mask = _get_subset_mask(agg,
                                   min_subset_size=min_subset_size,
                                   max_subset_size=max_subset_size,
                                   min_degree=min_degree,
                                   max_degree=max_degree,
                                   present=present, absent=absent)

    if subset_mask is True:
        return df, agg

    agg = agg[subset_mask]
    df = df[df.index.isin(agg.index)]
    return df, agg


class QueryResult:
    """Container for reformatted data and aggregates

    Attributes
    ----------
    data : DataFrame
        Selected samples. The index is a MultiIndex with one boolean level for
        each category.
    subsets : dict[frozenset, DataFrame]
        Dataframes for each intersection of categories.
    subset_sizes : Series
        Total size of each selected subset as a series. The index is as
        for `data`.
    category_totals : Series
        Total size of each category, regardless of selection.
    """
    def __init__(self, data, subset_sizes, category_totals):
        self.data = data
        self.subset_sizes = subset_sizes
        self.category_totals = category_totals

    def __repr__(self):
        return ("QueryResult(data={data}, subset_sizes={subset_sizes}, "
                "category_totals={category_totals}".format(**vars(self)))

    @property
    def subsets(self):
        categories = np.asarray(self.data.index.names)
        return {
            frozenset(categories.take(mask)): subset_data
            for mask, subset_data
            in self.data.groupby(level=list(range(len(categories))),
                                 sort=False)
        }


def query(data, present=None, absent=None,
          min_subset_size=None, max_subset_size=None,
          min_degree=None, max_degree=None,
          sort_by='degree', sort_categories_by='cardinality',
          subset_size='auto', sum_over=None, include_empty_subsets=False):
    """Transform and filter a categorised dataset

    Retrieve the set of items and totals corresponding to subsets of interest.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Elements associated with categories (a DataFrame), or the size of each
        subset of categories (a Series).
        Should have MultiIndex where each level is binary,
        corresponding to category membership.
        If a DataFrame, `sum_over` must be a string or False.
    present : str or list of str, optional
        Category or categories that must be present in subsets for styling.
    absent : str or list of str, optional
        Category or categories that must not be present in subsets for
        styling.
    min_subset_size : int, optional
        Minimum size of a subset to be reported. All subsets with
        a size smaller than this threshold will be omitted from
        category_totals and data.
        Size may be a sum of values, see `subset_size`.
    max_subset_size : int, optional
        Maximum size of a subset to be reported.
    min_degree : int, optional
        Minimum degree of a subset to be reported.
    max_degree : int, optional
        Maximum degree of a subset to be reported.
    sort_by : {'cardinality', 'degree', '-cardinality', '-degree',
               'input', '-input'}
        If 'cardinality', subset are listed from largest to smallest.
        If 'degree', they are listed in order of the number of categories
        intersected. If 'input', the order they appear in the data input is
        used.
        Prefix with '-' to reverse the ordering.

        Note this affects ``subset_sizes`` but not ``data``.
    sort_categories_by : {'cardinality', '-cardinality', 'input', '-input'}
        Whether to sort the categories by total cardinality, or leave them
        in the input data's provided order (order of index levels).
        Prefix with '-' to reverse the ordering.
    subset_size : {'auto', 'count', 'sum'}
        Configures how to calculate the size of a subset. Choices are:

        'auto' (default)
            If `data` is a DataFrame, count the number of rows in each group,
            unless `sum_over` is specified.
            If `data` is a Series with at most one row for each group, use
            the value of the Series. If `data` is a Series with more than one
            row per group, raise a ValueError.
        'count'
            Count the number of rows in each group.
        'sum'
            Sum the value of the `data` Series, or the DataFrame field
            specified by `sum_over`.
    sum_over : str or None
        If `subset_size='sum'` or `'auto'`, then the intersection size is the
        sum of the specified field in the `data` DataFrame. If a Series, only
        None is supported and its value is summed.
    include_empty_subsets : bool (default=False)
        If True, all possible category combinations will be returned in
        subset_sizes, even when some are not present in data.

    Returns
    -------
    QueryResult
        Including filtered ``data``, filtered and sorted ``subset_sizes`` and
        overall ``category_totals``.

    Examples
    --------
    >>> from upsetplot import query, generate_samples
    >>> data = generate_samples(n_samples=20)
    >>> result = query(data, present="cat1", max_subset_size=4)
    >>> result.category_totals
    cat1    14
    cat2     4
    cat0     0
    dtype: int64
    >>> result.subset_sizes
    cat1  cat2  cat0
    True  True  False    3
    Name: size, dtype: int64
    >>> result.data
                     index     value
    cat1 cat2 cat0
    True True False      0  2.04...
              False      2  2.05...
              False     10  2.55...
    >>>
    >>> # Sorting:
    >>> query(data, min_degree=1, sort_by="degree").subset_sizes
    cat1   cat2   cat0
    True   False  False    11
    False  True   False     1
    True   True   False     3
    Name: size, dtype: int64
    >>> query(data, min_degree=1, sort_by="cardinality").subset_sizes
    cat1   cat2   cat0
    True   False  False    11
           True   False     3
    False  True   False     1
    Name: size, dtype: int64
    >>>
    >>> # Getting each subset's data
    >>> result = query(data)
    >>> result.subsets[frozenset({"cat1", "cat2"})]
                index     value
    cat1  cat2 cat0
    False True False      3  1.333795
    >>> result.subsets[frozenset({"cat1"})]
                        index     value
    cat1  cat2  cat0
    False False False      5  0.918174
                False      8  1.948521
                False      9  1.086599
                False     13  1.105696
                False     19  1.339895
    """

    data, agg = _aggregate_data(data, subset_size, sum_over)
    data = _check_index(data)
    totals = [agg[agg.index.get_level_values(name).values.astype(bool)].sum()
              for name in agg.index.names]
    totals = pd.Series(totals, index=agg.index.names)

    if include_empty_subsets:
        nlevels = len(agg.index.levels)
        if nlevels > 10:
            raise ValueError(
                "include_empty_subsets is supported for at most 10 categories")
        new_agg = pd.Series(0,
                            index=pd.MultiIndex.from_product(
                                [[False, True]] * nlevels,
                                names=agg.index.names),
                            dtype=agg.dtype,
                            name=agg.name)
        new_agg.update(agg)
        agg = new_agg

    data, agg = _filter_subsets(data, agg,
                                min_subset_size=min_subset_size,
                                max_subset_size=max_subset_size,
                                min_degree=min_degree,
                                max_degree=max_degree,
                                present=present, absent=absent)

    # sort:
    if sort_categories_by in ('cardinality', '-cardinality'):
        totals.sort_values(ascending=sort_categories_by[:1] == '-',
                           inplace=True)
    elif sort_categories_by == '-input':
        totals = totals[::-1]
    elif sort_categories_by in (None, 'input'):
        pass
    else:
        raise ValueError('Unknown sort_categories_by: %r' % sort_categories_by)
    data = data.reorder_levels(totals.index.values)
    agg = agg.reorder_levels(totals.index.values)

    if sort_by in ('cardinality', '-cardinality'):
        agg = agg.sort_values(ascending=sort_by[:1] == '-')
    elif sort_by in ('degree', '-degree'):
        index_tuples = sorted(agg.index,
                              key=lambda x: (sum(x),) + tuple(reversed(x)),
                              reverse=sort_by[:1] == '-')
        agg = agg.reindex(pd.MultiIndex.from_tuples(index_tuples,
                                                    names=agg.index.names))
    elif sort_by == '-input':
        print("<", agg)
        agg = agg[::-1]
        print(">", agg)
    elif sort_by in (None, 'input'):
        pass
    else:
        raise ValueError('Unknown sort_by: %r' % sort_by)

    return QueryResult(data=data, subset_sizes=agg, category_totals=totals)
