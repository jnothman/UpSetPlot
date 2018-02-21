import pytest
from pandas.util.testing import assert_series_equal
import numpy as np

from upsetplot.plotting import _process_data
from upsetplot.data import generate_data


def is_ascending(seq):
    # return np.all(np.diff(seq) >= 0)
    return sorted(seq) == list(seq)


@pytest.mark.parametrize('X', [
    generate_data(aggregated=True),
    generate_data(aggregated=True).iloc[1:-2],
])
@pytest.mark.parametrize('order', ['size', 'degree'])
@pytest.mark.parametrize('order_categories', [False, True])
def test_process_data(X, order, order_categories):
    intersections, totals = _process_data(X,
                                          order=order,
                                          order_categories=order_categories)
    X_reordered = (X
                   .reorder_levels(intersections.index.names)
                   .reindex(index=intersections.index))
    assert_series_equal(X_reordered, intersections,
                        check_dtype=False)

    if order == 'size':
        assert is_ascending(intersections.values[::-1])
    else:
        # check degree order
        assert is_ascending(intersections.index.to_frame().sum(axis=1))
        # TODO: within a same-degree group, the tuple of active names should
        #       be in sort-order
    if order_categories:
        assert is_ascending(totals.values[::-1])

    assert np.all(totals.index.values == intersections.index.names)


@pytest.mark.parametrize('order', ['size', 'degree'])
@pytest.mark.parametrize('order_categories', [False, True])
def test_not_aggregated(order, order_categories):
    # FIXME: this is not testing if aggregation used is count or sum
    kw = {'order': order, 'order_categories': order_categories}
    Xagg = generate_data(aggregated=True)
    intersections1, totals1 = _process_data(Xagg, **kw)
    Xunagg = generate_data()
    Xunagg.loc[:] = 1
    intersections2, totals2 = _process_data(Xunagg, **kw)
    assert_series_equal(intersections1, intersections2,
                        check_dtype=False)
    assert_series_equal(totals1, totals2, check_dtype=False)
