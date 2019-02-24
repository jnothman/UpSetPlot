import io
import itertools

import pytest
from pandas.util.testing import assert_series_equal
import pandas as pd
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt

from upsetplot import plot
from upsetplot import UpSet
from upsetplot import generate_data
from upsetplot.plotting import _process_data


def is_ascending(seq):
    # return np.all(np.diff(seq) >= 0)
    return sorted(seq) == list(seq)


@pytest.mark.parametrize('X', [
    generate_data(aggregated=True),
    generate_data(aggregated=True).iloc[1:-2],
])
@pytest.mark.parametrize('sort_by', ['cardinality', 'degree'])
@pytest.mark.parametrize('sort_sets_by', [None, 'cardinality'])
def test_process_data(X, sort_by, sort_sets_by):
    intersections, totals = _process_data(X,
                                          sort_by=sort_by,
                                          sort_sets_by=sort_sets_by)
    X_reordered = (X
                   .reorder_levels(intersections.index.names)
                   .reindex(index=intersections.index))
    assert len(X) == len(X_reordered)
    assert X_reordered.index.is_unique
    assert_series_equal(X_reordered, intersections,
                        check_dtype=False)

    if sort_by == 'cardinality':
        assert is_ascending(intersections.values[::-1])
    else:
        # check degree order
        assert is_ascending(intersections.index.to_frame().sum(axis=1))
        # TODO: within a same-degree group, the tuple of active names should
        #       be in sort-order
    if sort_sets_by:
        assert is_ascending(totals.values[::-1])

    assert np.all(totals.index.values == intersections.index.names)


@pytest.mark.parametrize('sort_by', ['cardinality', 'degree'])
@pytest.mark.parametrize('sort_sets_by', [None, 'cardinality'])
def test_not_aggregated(sort_by, sort_sets_by):
    # FIXME: this is not testing if aggregation used is count or sum
    kw = {'sort_by': sort_by, 'sort_sets_by': sort_sets_by}
    Xagg = generate_data(aggregated=True)
    intersections1, totals1 = _process_data(Xagg, **kw)
    Xunagg = generate_data()
    Xunagg.loc[:] = 1
    intersections2, totals2 = _process_data(Xunagg, **kw)
    assert_series_equal(intersections1, intersections2,
                        check_dtype=False)
    assert_series_equal(totals1, totals2, check_dtype=False)


@pytest.mark.parametrize('kw', [{'sort_by': 'blah'},
                                {'sort_by': True},
                                {'sort_by': None},
                                {'sort_sets_by': 'blah'},
                                {'sort_sets_by': True}])
def test_param_validation(kw):
    X = generate_data(n_samples=100)
    with pytest.raises(ValueError):
        UpSet(X, **kw)


@pytest.mark.parametrize('kw', [{},
                                {'element_size': None},
                                {'orientation': 'vertical'}])
def test_plot_smoke_test(kw):
    fig = matplotlib.figure.Figure()
    X = generate_data(n_samples=100)
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
         fig)


def test_dataframe_raises():
    fig = matplotlib.figure.Figure()
    df = pd.DataFrame({'val': [5, 7],
                       'set1': [False, True],
                       'set2': [True, True]}).set_index(['set1', 'set2'])
    with pytest.raises(ValueError, match='must be a pandas.Series'):
        plot(df, fig)


def test_vertical():
    X = generate_data(n_samples=100)

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
    X = generate_data(n_samples=100)
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
    X = generate_data(n_samples=100)
    plot(X, fig)
    n_artists_no_sizes = _count_descendants(fig)

    fig = matplotlib.figure.Figure()
    plot(X, fig, show_counts=True)
    n_artists_yes_sizes = _count_descendants(fig)
    assert n_artists_yes_sizes - n_artists_no_sizes > 6

    fig = matplotlib.figure.Figure()
    plot(X, fig, show_counts='%0.2g')
    assert n_artists_yes_sizes == _count_descendants(fig)

    with pytest.raises(ValueError):
        fig = matplotlib.figure.Figure()
        plot(X, fig, show_counts='%0.2h')
