from collections import OrderedDict
import pytest
import pandas as pd
import numpy as np
from distutils.version import LooseVersion
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_index_equal)
from upsetplot import from_memberships, from_contents


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


@pytest.mark.parametrize('data,ndim', [
    ([1, 2, 3, 4], 1),
    (np.array([1, 2, 3, 4]), 1),
    (pd.Series([1, 2, 3, 4], name='foo'), 1),
    ([[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']], 2),
    (pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c'], [4, 'd']],
                  columns=['foo', 'bar'],
                  index=['q', 'r', 's', 't']), 2),
])
def test_from_memberships_with_data(data, ndim):
    memberships = [[],
                   ['hello'],
                   ['world'],
                   ['hello', 'world']]
    out = from_memberships(memberships, data=data)
    assert out is not data  # make sure frame is copied
    if hasattr(data, 'loc') and np.asarray(data).dtype.kind in 'ifb':
        # but not deepcopied when possible
        if LooseVersion(pd.__version__) > LooseVersion('0.35'):
            assert out.values.base is np.asarray(data).base
    if ndim == 1:
        assert isinstance(out, pd.Series)
    else:
        assert isinstance(out, pd.DataFrame)
    assert_frame_equal(pd.DataFrame(out).reset_index(drop=True),
                       pd.DataFrame(data).reset_index(drop=True))
    no_data = from_memberships(memberships=memberships)
    assert_index_equal(out.index, no_data.index)

    with pytest.raises(ValueError, match='length'):
        from_memberships(memberships[:-1], data=data)


@pytest.mark.parametrize('typ', [set, list, tuple, iter])
@pytest.mark.parametrize('id_column', ['id', 'blah'])
def test_from_contents(typ, id_column):
    contents = {'cat1': typ(['aa', 'bb', 'cc']),
                'cat2': typ(['cc', 'dd']),
                'cat3': typ(['ee']),
                }
    empty_data = pd.DataFrame(index=['aa', 'bb', 'cc', 'dd', 'ee', 'ff'])
    out = from_contents(OrderedDict(contents), data=empty_data,
                        id_column=id_column)
    out2 = from_memberships(memberships=[{'cat1'},
                                         {'cat1'},
                                         {'cat1', 'cat2'},
                                         {'cat2'},
                                         {'cat3'},
                                         []],
                            data=empty_data)
    assert_series_equal(out[id_column].reset_index(drop=True),
                        pd.Series(['aa', 'bb', 'cc', 'dd', 'ee', 'ff'],
                                  name=id_column))
    assert_frame_equal(out.drop(columns=[id_column]), out2)

    # TODO: empty category (can't be represented with from_memberships)
    # TODO: unordered dict
    # TODO: check that you can have entries in data that are not in contents.
    # TODO: error cases
