import pytest
import pandas as pd
import numpy as np
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_index_equal)
from upsetplot import from_memberships


@pytest.mark.parametrize('typ', [set, list, tuple, iter])
def test_from_memberships_no_data(typ):
    with pytest.raises(ValueError, match='at least one set'):
        from_memberships([])
    with pytest.raises(ValueError, match='at least one set'):
        from_memberships([[], []])
    with pytest.raises(ValueError, match='strings'):
        from_memberships([[1]])
    with pytest.raises(ValueError, match='strings'):
        from_memberships([[1, 'str']])
    with pytest.raises(TypeError):
        from_memberships([1])

    out = from_memberships([typ([]),
                            typ(['hello']),
                            typ(['world']),
                            typ(['hello', 'world']),
                            ])
    exp = pd.DataFrame([[False, False, 1],
                        [True, False, 1],
                        [False, True, 1],
                        [True, True, 1]],
                       columns=['hello', 'world', 'ones']
                       ).set_index(['hello', 'world'])['ones']
    assert isinstance(exp.index, pd.MultiIndex)
    assert_series_equal(exp, out)

    # test sorting by name
    out = from_memberships([typ(['hello']),
                            typ(['world'])])
    exp = pd.DataFrame([[True, False, 1],
                        [False, True, 1]],
                       columns=['hello', 'world', 'ones']
                       ).set_index(['hello', 'world'])['ones']
    assert_series_equal(exp, out)
    out = from_memberships([typ(['world']),
                            typ(['hello'])])
    exp = pd.DataFrame([[False, True, 1],
                        [True, False, 1]],
                       columns=['hello', 'world', 'ones']
                       ).set_index(['hello', 'world'])['ones']
    assert_series_equal(exp, out)


@pytest.mark.parametrize('data', [
    [1, 2, 3, 4],
    np.array([1, 2, 3, 4]),
    pd.Series([1, 2, 3, 4], name='foo'),
    [[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']],
    pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']],
                 columns=['foo', 'bar'],
                 index=['q', 'r', 's', 't']),
])
def test_from_memberships_with_data(data):
    memberships = [[],
                   ['hello'],
                   ['world'],
                   ['hello', 'world']]
    out = from_memberships(memberships, data=data)
    assert out is not data  # make sure frame is copied
    if hasattr(data, 'loc') and np.asarray(data).dtype.kind in 'ifb':
        # but not deepcopied when possible
        assert out.values.base is np.asarray(data).base
    if isinstance(data, pd.Series):
        assert isinstance(out, pd.Series)
    else:
        assert isinstance(out, pd.DataFrame)
    assert_frame_equal(pd.DataFrame(out).reset_index(drop=True),
                       pd.DataFrame(data).reset_index(drop=True))
    no_data = from_memberships(memberships=memberships)
    assert_index_equal(out.index, no_data.index)

    with pytest.raises(ValueError, match='length'):
        from_memberships(memberships[:-1], data=data)
