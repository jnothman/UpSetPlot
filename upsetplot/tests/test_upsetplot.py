import io

import pytest
from pandas.util.testing import assert_series_equal
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


@pytest.mark.parametrize('kw', [{}, {'element_size': None}])
def test_plot_smoke_test(kw):
    fig = matplotlib.figure.Figure()
    X = generate_data(n_samples=100)
    plot(X, fig, **kw)
    fig.savefig(io.BytesIO(), format='png')

    # Also check fig is optional
    n_nums = len(plt.get_fignums())
    plot(X)
    assert len(plt.get_fignums()) - n_nums == 1
    assert plt.gcf().axes


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
