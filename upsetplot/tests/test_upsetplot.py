import io
import itertools

import pytest
from pandas.util.testing import (
    assert_series_equal, assert_frame_equal, assert_index_equal)
from numpy.testing import assert_array_equal
import pandas as pd
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.text import Text

from upsetplot import plot
from upsetplot import UpSet
from upsetplot import generate_counts, generate_samples
from upsetplot.plotting import _process_data

# TODO: warnings should raise errors


def is_ascending(seq):
    # return np.all(np.diff(seq) >= 0)
    return sorted(seq) == list(seq)


def get_all_texts(mpl_artist):
    out = [text.get_text() for text in mpl_artist.findobj(Text)]
    return [text for text in out if text]


@pytest.mark.parametrize('x', [
    generate_counts(),
    generate_counts().iloc[1:-2],
])
@pytest.mark.parametrize('sort_by', ['cardinality', 'degree'])
@pytest.mark.parametrize('sort_categories_by', [None, 'cardinality'])
def test_process_data_series(x, sort_by, sort_categories_by):
    assert x.name == 'value'
    for subset_size in ['auto', 'sum', 'count']:
        for sum_over in ['abc', False]:
            with pytest.raises(ValueError, match='sum_over is not applicable'):
                _process_data(x, sort_by=sort_by,
                              sort_categories_by=sort_categories_by,
                              subset_size=subset_size, sum_over=sum_over)

    total, df, intersections, totals = _process_data(
        x, subset_size='auto', sort_by=sort_by,
        sort_categories_by=sort_categories_by, sum_over=None)

    assert total == x.sum()

    assert intersections.name == 'value'
    x_reordered = (x
                   .reorder_levels(intersections.index.names)
                   .reindex(index=intersections.index))
    assert len(x) == len(x_reordered)
    assert x_reordered.index.is_unique
    assert_series_equal(x_reordered, intersections,
                        check_dtype=False)

    if sort_by == 'cardinality':
        assert is_ascending(intersections.values[::-1])
    else:
        # check degree order
        assert is_ascending(intersections.index.to_frame().sum(axis=1))
        # TODO: within a same-degree group, the tuple of active names should
        #       be in sort-order
    if sort_categories_by:
        assert is_ascending(totals.values[::-1])

    assert np.all(totals.index.values == intersections.index.names)

    assert np.all(df.index.names == intersections.index.names)
    assert set(df.columns) == {'_value', '_bin'}
    assert_index_equal(df['_value'].reorder_levels(x.index.names).index,
                       x.index)
    assert_array_equal(df['_value'], x)
    assert_index_equal(intersections.iloc[df['_bin']].index,
                       df.index)
    assert len(df) == len(x)


@pytest.mark.parametrize('x', [
    generate_samples()['value'],
    generate_counts(),
])
def test_subset_size_series(x):
    kw = {'sort_by': 'cardinality',
          'sort_categories_by': 'cardinality',
          'sum_over': None}
    total, df_sum, intersections_sum, totals_sum = _process_data(
        x, subset_size='sum', **kw)
    assert total == intersections_sum.sum()

    if x.index.is_unique:
        total, df, intersections, totals = _process_data(
            x, subset_size='auto', **kw)
        assert total == intersections.sum()
        assert_frame_equal(df, df_sum)
        assert_series_equal(intersections, intersections_sum)
        assert_series_equal(totals, totals_sum)
    else:
        with pytest.raises(ValueError):
            _process_data(x, subset_size='auto', **kw)

    total, df_count, intersections_count, totals_count = _process_data(
        x, subset_size='count', **kw)
    assert total == intersections_count.sum()
    total, df, intersections, totals = _process_data(
        x.groupby(level=list(range(len(x.index.levels)))).count(),
        subset_size='sum', **kw)
    assert total == intersections.sum()
    assert_series_equal(intersections, intersections_count, check_names=False)
    assert_series_equal(totals, totals_count)


@pytest.mark.parametrize('x', [
    generate_samples()['value'],
])
@pytest.mark.parametrize('sort_by', ['cardinality', 'degree'])
@pytest.mark.parametrize('sort_categories_by', [None, 'cardinality'])
def test_process_data_frame(x, sort_by, sort_categories_by):
    X = pd.DataFrame({'a': x})

    with pytest.warns(None):
        total, df, intersections, totals = _process_data(
            X, sort_by=sort_by, sort_categories_by=sort_categories_by,
            sum_over='a', subset_size='auto')
    assert df is not X
    assert total == intersections.sum()

    # check equivalence to Series
    total1, df1, intersections1, totals1 = _process_data(
        x, sort_by=sort_by, sort_categories_by=sort_categories_by,
        subset_size='sum', sum_over=None)

    assert intersections.name == 'a'
    assert_frame_equal(df, df1.rename(columns={'_value': 'a'}))
    assert_series_equal(intersections, intersections1, check_names=False)
    assert_series_equal(totals, totals1)

    # check effect of extra column
    X = pd.DataFrame({'a': x, 'b': np.arange(len(x))})
    total2, df2, intersections2, totals2 = _process_data(
        X, sort_by=sort_by, sort_categories_by=sort_categories_by,
        sum_over='a', subset_size='auto')
    assert total2 == intersections2.sum()
    assert_series_equal(intersections, intersections2)
    assert_series_equal(totals, totals2)
    assert_frame_equal(df, df2.drop('b', axis=1))
    assert_array_equal(df2['b'], X['b'])  # disregard levels, tested above

    # check effect not dependent on order/name
    X = pd.DataFrame({'b': np.arange(len(x)), 'c': x})
    total3, df3, intersections3, totals3 = _process_data(
        X, sort_by=sort_by, sort_categories_by=sort_categories_by,
        sum_over='c', subset_size='auto')
    assert total3 == intersections3.sum()
    assert_series_equal(intersections, intersections3, check_names=False)
    assert intersections.name == 'a'
    assert intersections3.name == 'c'
    assert_series_equal(totals, totals3)
    assert_frame_equal(df.rename(columns={'a': 'c'}), df3.drop('b', axis=1))
    assert_array_equal(df3['b'], X['b'])

    # check subset_size='count'
    X = pd.DataFrame({'b': np.ones(len(x), dtype=int), 'c': x})
    total4, df4, intersections4, totals4 = _process_data(
        X, sort_by=sort_by, sort_categories_by=sort_categories_by,
        sum_over='b', subset_size='auto')
    total5, df5, intersections5, totals5 = _process_data(
        X, sort_by=sort_by, sort_categories_by=sort_categories_by,
        subset_size='count', sum_over=None)
    assert total5 == intersections5.sum()
    assert_series_equal(intersections4, intersections5, check_names=False)
    assert intersections4.name == 'b'
    assert intersections5.name == 'size'
    assert_series_equal(totals4, totals5)
    assert_frame_equal(df4, df5)


@pytest.mark.parametrize('x', [
    generate_samples()['value'],
    generate_counts(),
])
def test_subset_size_frame(x):
    kw = {'sort_by': 'cardinality',
          'sort_categories_by': 'cardinality'}
    X = pd.DataFrame({'x': x})
    total_sum, df_sum, intersections_sum, totals_sum = _process_data(
        X, subset_size='sum', sum_over='x', **kw)
    total_count, df_count, intersections_count, totals_count = _process_data(
        X, subset_size='count', sum_over=None, **kw)

    # error cases: sum_over=False
    for subset_size in ['auto', 'sum', 'count']:
        with pytest.raises(ValueError, match='sum_over'):
            _process_data(
                X, subset_size=subset_size, sum_over=False, **kw)

    with pytest.raises(ValueError, match='sum_over'):
        _process_data(
            X, subset_size=subset_size, sum_over=False, **kw)

    # error cases: sum_over incompatible with subset_size
    with pytest.raises(ValueError, match='sum_over should be a field'):
        _process_data(
            X, subset_size='sum', sum_over=None, **kw)
    with pytest.raises(ValueError, match='sum_over cannot be set'):
        _process_data(
            X, subset_size='count', sum_over='x', **kw)

    # check subset_size='auto' with sum_over=str => sum
    total, df, intersections, totals = _process_data(
        X, subset_size='auto', sum_over='x', **kw)
    assert total == intersections.sum()
    assert_frame_equal(df, df_sum)
    assert_series_equal(intersections, intersections_sum)
    assert_series_equal(totals, totals_sum)

    # check subset_size='auto' with sum_over=None => count
    total, df, intersections, totals = _process_data(
        X, subset_size='auto', sum_over=None, **kw)
    assert total == intersections.sum()
    assert_frame_equal(df, df_count)
    assert_series_equal(intersections, intersections_count)
    assert_series_equal(totals, totals_count)


@pytest.mark.parametrize('sort_by', ['cardinality', 'degree'])
@pytest.mark.parametrize('sort_categories_by', [None, 'cardinality'])
def test_not_unique(sort_by, sort_categories_by):
    kw = {'sort_by': sort_by,
          'sort_categories_by': sort_categories_by,
          'subset_size': 'sum',
          'sum_over': None}
    Xagg = generate_counts()
    total1, df1, intersections1, totals1 = _process_data(Xagg, **kw)
    Xunagg = generate_samples()['value']
    Xunagg.loc[:] = 1
    total2, df2, intersections2, totals2 = _process_data(Xunagg, **kw)
    assert_series_equal(intersections1, intersections2,
                        check_dtype=False)
    assert total2 == intersections2.sum()
    assert_series_equal(totals1, totals2, check_dtype=False)
    assert set(df1.columns) == {'_value', '_bin'}
    assert set(df2.columns) == {'_value', '_bin'}
    assert len(df2) == len(Xunagg)
    assert df2['_bin'].nunique() == len(intersections2)


@pytest.mark.parametrize('kw', [{'sort_by': 'blah'},
                                {'sort_by': True},
                                {'sort_by': None},
                                {'sort_categories_by': 'blah'},
                                {'sort_categories_by': True}])
def test_param_validation(kw):
    X = generate_counts(n_samples=100)
    with pytest.raises(ValueError):
        UpSet(X, **kw)


@pytest.mark.parametrize('kw', [{},
                                {'element_size': None},
                                {'orientation': 'vertical'},
                                {'intersection_plot_elements': 0}])
def test_plot_smoke_test(kw):
    fig = matplotlib.figure.Figure()
    X = generate_counts(n_samples=100)
    plot(X, fig, **kw)
    fig.savefig(io.BytesIO(), format='png')

    # Also check fig is optional
    n_nums = len(plt.get_fignums())
    plot(X, **kw)
    assert len(plt.get_fignums()) - n_nums == 1
    assert plt.gcf().axes


@pytest.mark.parametrize('set1',
                         itertools.product([False, True], repeat=2))
@pytest.mark.parametrize('set2',
                         itertools.product([False, True], repeat=2))
def test_two_sets(set1, set2):
    # we had a bug where processing failed if no items were in some set
    fig = matplotlib.figure.Figure()
    plot(pd.DataFrame({'val': [5, 7],
                       'set1': set1,
                       'set2': set2}).set_index(['set1', 'set2'])['val'],
         fig, subset_size='sum')


def test_vertical():
    X = generate_counts(n_samples=100)

    fig = matplotlib.figure.Figure()
    UpSet(X, orientation='horizontal').make_grid(fig)
    horz_height = fig.get_figheight()
    horz_width = fig.get_figwidth()
    assert horz_height < horz_width

    fig = matplotlib.figure.Figure()
    UpSet(X, orientation='vertical').make_grid(fig)
    vert_height = fig.get_figheight()
    vert_width = fig.get_figwidth()
    assert horz_width / horz_height > vert_width / vert_height

    # TODO: test axes positions, plot order, bar orientation
    pass


def test_element_size():
    X = generate_counts(n_samples=100)
    figsizes = []
    for element_size in range(10, 50, 5):
        fig = matplotlib.figure.Figure()
        UpSet(X, element_size=element_size).make_grid(fig)
        figsizes.append((fig.get_figwidth(), fig.get_figheight()))

    figwidths, figheights = zip(*figsizes)
    # Absolute width increases
    assert np.all(np.diff(figwidths) > 0)
    aspect = np.divide(figwidths, figheights)
    # Font size stays constant, so aspect ratio decreases
    assert np.all(np.diff(aspect) < 0)
    # But doesn't decrease by much
    assert np.all(aspect[:-1] / aspect[1:] < 1.1)

    fig = matplotlib.figure.Figure()
    figsize_before = fig.get_figwidth(), fig.get_figheight()
    UpSet(X, element_size=None).make_grid(fig)
    figsize_after = fig.get_figwidth(), fig.get_figheight()
    assert figsize_before == figsize_after

    # TODO: make sure axes are all within figure


def _walk_artists(el):
    children = el.get_children()
    yield el, children
    for ch in children:
        for x in _walk_artists(ch):
            yield x


def _count_descendants(el):
    return sum(len(children) for x, children in _walk_artists(el))


@pytest.mark.parametrize('orientation', ['horizontal', 'vertical'])
def test_show_counts(orientation):
    fig = matplotlib.figure.Figure()
    X = generate_counts(n_samples=10000)
    plot(X, fig, orientation=orientation)
    n_artists_no_sizes = _count_descendants(fig)

    fig = matplotlib.figure.Figure()
    plot(X, fig, orientation=orientation, show_counts=True)
    n_artists_yes_sizes = _count_descendants(fig)
    assert n_artists_yes_sizes - n_artists_no_sizes > 6
    assert '9547' in get_all_texts(fig)  # set size
    assert '283' in get_all_texts(fig)   # intersection size

    fig = matplotlib.figure.Figure()
    plot(X, fig, orientation=orientation, show_counts='%0.2g')
    assert n_artists_yes_sizes == _count_descendants(fig)
    assert '9.5e+03' in get_all_texts(fig)
    assert '2.8e+02' in get_all_texts(fig)

    fig = matplotlib.figure.Figure()
    plot(X, fig, orientation=orientation, show_percentages=True)
    assert n_artists_yes_sizes == _count_descendants(fig)
    assert '95.5%' in get_all_texts(fig)
    assert '2.8%' in get_all_texts(fig)

    fig = matplotlib.figure.Figure()
    plot(X, fig, orientation=orientation, show_counts=True,
         show_percentages=True)
    assert n_artists_yes_sizes == _count_descendants(fig)
    if orientation == 'vertical':
        assert '9547\n(95.5%)' in get_all_texts(fig)
        assert '283 (2.8%)' in get_all_texts(fig)
    else:
        assert '9547 (95.5%)' in get_all_texts(fig)
        assert '283\n(2.8%)' in get_all_texts(fig)

    with pytest.raises(ValueError):
        fig = matplotlib.figure.Figure()
        plot(X, fig, orientation=orientation, show_counts='%0.2h')


def test_add_catplot():
    pytest.importorskip('seaborn')
    X = generate_counts(n_samples=100)
    upset = UpSet(X)
    # smoke test
    upset.add_catplot('violin')
    fig = matplotlib.figure.Figure()
    upset.plot(fig)

    # can't provide value with Series
    with pytest.raises(ValueError):
        upset.add_catplot('violin', value='foo')

    # check the above add_catplot did not break the state
    upset.plot(fig)

    X = generate_counts(n_samples=100)
    X.name = 'foo'
    X = X.to_frame()
    upset = UpSet(X, subset_size='count')
    # must provide value with DataFrame
    with pytest.raises(ValueError):
        upset.add_catplot('violin')
    upset.add_catplot('violin', value='foo')
    with pytest.raises(ValueError):
        # not a known column
        upset.add_catplot('violin', value='bar')
    upset.plot(fig)

    # invalid plot kind raises error when plotting
    upset.add_catplot('foobar', value='foo')
    with pytest.raises(AttributeError):
        upset.plot(fig)


@pytest.mark.parametrize('x', [
    generate_counts(),
])
def test_index_must_be_bool(x):
    # Truthy ints are okay
    x = x.reset_index()
    x[['cat0', 'cat2', 'cat2']] = x[['cat0', 'cat1', 'cat2']].astype(int)
    x = x.set_index(['cat0', 'cat1', 'cat2']).iloc[:, 0]

    UpSet(x)

    # other ints are not
    x = x.reset_index()
    x[['cat0', 'cat2', 'cat2']] = x[['cat0', 'cat1', 'cat2']] + 1
    x = x.set_index(['cat0', 'cat1', 'cat2']).iloc[:, 0]
    with pytest.raises(ValueError, match='not boolean'):
        UpSet(x)
