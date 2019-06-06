from collections import OrderedDict
import pytest
import pandas as pd
import numpy as np
from distutils.version import LooseVersion
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_index_equal)
from upsetplot import from_memberships, from_contents, generate_data


@pytest.mark.parametrize('typ', [set, list, tuple, iter])
def test_from_memberships_no_data(typ):
    with pytest.raises(ValueError, match='at least one category'):
        from_memberships([])
    with pytest.raises(ValueError, match='at least one category'):
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


@pytest.mark.parametrize('data', [None,
                                  {'attr1': [3, 4, 5, 6, 7, 8],
                                   'attr2': list('qrstuv')}])
@pytest.mark.parametrize('typ', [set, list, tuple, iter])
@pytest.mark.parametrize('id_column', ['id', 'blah'])
def test_from_contents_vs_memberships(data, typ, id_column):
    contents = OrderedDict([('cat1', typ(['aa', 'bb', 'cc'])),
                            ('cat2', typ(['cc', 'dd'])),
                            ('cat3', typ(['ee']))])
    # Note that ff is not present in contents
    data_df = pd.DataFrame(data,
                           index=['aa', 'bb', 'cc', 'dd', 'ee', 'ff'])
    baseline = from_contents(contents, data=data_df,
                             id_column=id_column)
    # compare from_contents to from_memberships
    expected = from_memberships(memberships=[{'cat1'},
                                             {'cat1'},
                                             {'cat1', 'cat2'},
                                             {'cat2'},
                                             {'cat3'},
                                             []],
                                data=data_df)
    assert_series_equal(baseline[id_column].reset_index(drop=True),
                        pd.Series(['aa', 'bb', 'cc', 'dd', 'ee', 'ff'],
                                  name=id_column))
    assert_frame_equal(baseline.drop([id_column], axis=1), expected)


def test_from_contents(typ=set, id_column='id'):
    contents = OrderedDict([('cat1', {'aa', 'bb', 'cc'}),
                            ('cat2', {'cc', 'dd'}),
                            ('cat3', {'ee'})])
    empty_data = pd.DataFrame(index=['aa', 'bb', 'cc', 'dd', 'ee'])
    baseline = from_contents(contents, data=empty_data,
                             id_column=id_column)
    # data=None
    out = from_contents(contents, id_column=id_column)
    assert_frame_equal(out.sort_values(id_column), baseline)

    # unordered contents dict
    out = from_contents({'cat3': contents['cat3'],
                         'cat2': contents['cat2'],
                         'cat1': contents['cat1']},
                        data=empty_data, id_column=id_column)
    assert_frame_equal(out.reorder_levels(['cat1', 'cat2', 'cat3']),
                       baseline)

    # empty category
    out = from_contents({'cat1': contents['cat1'],
                         'cat2': contents['cat2'],
                         'cat3': contents['cat3'],
                         'cat4': []},
                        data=empty_data,
                        id_column=id_column)
    assert not out.index.to_frame()['cat4'].any()  # cat4 should be all-false
    assert len(out.index.names) == 4
    out.index = out.index.to_frame().set_index(['cat1', 'cat2', 'cat3']).index
    assert_frame_equal(out, baseline)


@pytest.mark.parametrize('id_column', ['id', 'blah'])
def test_from_contents_invalid(id_column):
    contents = OrderedDict([('cat1', {'aa', 'bb', 'cc'}),
                            ('cat2', {'cc', 'dd'}),
                            ('cat3', {'ee'})])
    with pytest.raises(ValueError, match='columns overlap'):
        from_contents(contents,
                      data=pd.DataFrame({'cat1': [1, 2, 3, 4, 5]}),
                      id_column=id_column)
    with pytest.raises(ValueError, match='duplicate ids'):
        from_contents({'cat1': ['aa', 'bb'],
                       'cat2': ['dd', 'dd']}, id_column=id_column)
    # category named id
    with pytest.raises(ValueError, match='cannot be named'):
        from_contents({id_column: {'aa', 'bb', 'cc'},
                       'cat2': {'cc', 'dd'},
                       }, id_column=id_column)
    # category named id
    with pytest.raises(ValueError, match='cannot contain'):
        from_contents(contents,
                      data=pd.DataFrame({id_column: [1, 2, 3, 4, 5]},
                                        index=['aa', 'bb', 'cc', 'dd', 'ee']),
                      id_column=id_column)
    with pytest.raises(ValueError, match='identifiers in contents'):
        from_contents({'cat1': ['aa']},
                      data=pd.DataFrame([[1]]),
                      id_column=id_column)


def test_generate_data_warning():
    with pytest.warns(DeprecationWarning):
        generate_data()
