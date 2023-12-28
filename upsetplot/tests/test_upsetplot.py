import io
import itertools

import pytest
from pandas.testing import (
    assert_series_equal, assert_frame_equal, assert_index_equal)
from numpy.testing import assert_array_equal
import pandas as pd
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.colors import to_hex
from matplotlib import cm

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
@pytest.mark.parametrize(
    'sort_by',
    ['cardinality', 'degree', '-cardinality', '-degree', None,
     'input', '-input'])
@pytest.mark.parametrize(
    'sort_categories_by',
    [None, 'input', '-input', 'cardinality', '-cardinality'])
def test_process_data_series(x, sort_by, sort_categories_by):
    assert x.name == 'value'
    for subset_size in ['auto', 'sum', 'count']:
        for sum_over in ['abc', False]:
            with pytest.raises(ValueError, match='sum_over is not applicable'):
                _process_data(x, sort_by=sort_by,
                              sort_categories_by=sort_categories_by,
                              subset_size=subset_size, sum_over=sum_over)

    # shuffle input to test sorting
    x = x.sample(frac=1., replace=False, random_state=0)

    total, df, intersections, totals = _process_data(
        x, subset_size='auto', sort_by=sort_by,
        sort_categories_by=sort_categories_by, sum_over=None)

    assert total == x.sum()

    assert intersections.name == 'value'
    x_reordered_levels = (x
                          .reorder_levels(intersections.index.names))
    x_reordered = (x_reordered_levels
                   .reindex(index=intersections.index))
    assert len(x) == len(x_reordered)
    assert x_reordered.index.is_unique
    assert_series_equal(x_reordered, intersections,
                        check_dtype=False)

    if sort_by == 'cardinality':
        assert is_ascending(intersections.values[::-1])
    elif sort_by == '-cardinality':
        assert is_ascending(intersections.values)
    elif sort_by == 'degree':
        # check degree order
        assert is_ascending(intersections.index.to_frame().sum(axis=1))
        # TODO: within a same-degree group, the tuple of active names should
        #       be in sort-order
    elif sort_by == '-degree':
        # check degree order
        assert is_ascending(intersections.index.to_frame().sum(axis=1)[::-1])
    else:
        find_first_in_orig = x_reordered_levels.index.tolist().index
        orig_order = [find_first_in_orig(key)
                      for key in intersections.index.tolist()]
        assert orig_order == sorted(
            orig_order,
            reverse=sort_by is not None and sort_by.startswith('-'))

    if sort_categories_by == 'cardinality':
        assert is_ascending(totals.values[::-1])
    elif sort_categories_by == '-cardinality':
        assert is_ascending(totals.values)

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
@pytest.mark.parametrize('sort_by', ['cardinality', 'degree', None])
@pytest.mark.parametrize('sort_categories_by', [None, 'cardinality'])
def test_process_data_frame(x, sort_by, sort_categories_by):
    # shuffle input to test sorting
    x = x.sample(frac=1., replace=False, random_state=0)

    X = pd.DataFrame({'a': x})

    with pytest.warns(None):
        total, df, intersections, totals = _process_data(
            X, sort_by=sort_by, sort_categories_by=sort_categories_by,
            sum_over='a', subset_size='auto')
    assert df is not X
    assert total == pytest.approx(intersections.sum())

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
    assert total2 == pytest.approx(intersections2.sum())
    assert_series_equal(intersections, intersections2)
    assert_series_equal(totals, totals2)
    assert_frame_equal(df, df2.drop('b', axis=1))
    assert_array_equal(df2['b'], X['b'])  # disregard levels, tested above

    # check effect not dependent on order/name
    X = pd.DataFrame({'b': np.arange(len(x)), 'c': x})
    total3, df3, intersections3, totals3 = _process_data(
        X, sort_by=sort_by, sort_categories_by=sort_categories_by,
        sum_over='c', subset_size='auto')
    assert total3 == pytest.approx(intersections3.sum())
    assert_series_equal(intersections, intersections3, check_names=False)
    assert intersections.name == 'a'
    assert intersections3.name == 'c'
    assert_series_equal(totals, totals3)
    assert_frame_equal(df.rename(columns={'a': 'c'}), df3.drop('b', axis=1))
    assert_array_equal(df3['b'], X['b'])

    # check subset_size='count'
    X = pd.DataFrame({'b': np.ones(len(x), dtype='int64'), 'c': x})

    total4, df4, intersections4, totals4 = _process_data(
        X, sort_by=sort_by, sort_categories_by=sort_categories_by,
        sum_over='b', subset_size='auto')
    total5, df5, intersections5, totals5 = _process_data(
        X, sort_by=sort_by, sort_categories_by=sort_categories_by,
        subset_size='count', sum_over=None)
    assert total5 == pytest.approx(intersections5.sum())
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


def test_include_empty_subsets():
    X = generate_counts(n_samples=2, n_categories=3)

    no_empty_upset = UpSet(X, include_empty_subsets=False)
    assert len(no_empty_upset.intersections) <= 2

    include_empty_upset = UpSet(X, include_empty_subsets=True)
    assert len(include_empty_upset.intersections) == 2 ** 3
    common_intersections = include_empty_upset.intersections.loc[
        no_empty_upset.intersections.index]
    assert_series_equal(no_empty_upset.intersections,
                        common_intersections)
    include_empty_upset.plot()  # smoke test


@pytest.mark.parametrize('kw', [{'sort_by': 'blah'},
                                {'sort_by': True},
                                {'sort_categories_by': 'blah'},
                                {'sort_categories_by': True}])
def test_param_validation(kw):
    X = generate_counts(n_samples=100)
    with pytest.raises(ValueError):
        UpSet(X, **kw)


@pytest.mark.parametrize('kw', [{},
                                {'element_size': None},
                                {'orientation': 'vertical'},
                                {'intersection_plot_elements': 0},
                                {'facecolor': 'red'},
                                {'shading_color': 'lightgrey',
                                 'other_dots_color': 'pink'}])
def test_plot_smoke_test(kw):
    fig = matplotlib.figure.Figure()
    X = generate_counts(n_samples=100)
    axes = plot(X, fig, **kw)
    fig.savefig(io.BytesIO(), format='png')

    attr = ('get_xlim'
            if kw.get('orientation', 'horizontal') == 'horizontal'
            else 'get_ylim')
    lim = getattr(axes['matrix'], attr)()
    expected_width = len(X)
    assert expected_width == lim[1] - lim[0]

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
    assert np.all(np.diff(aspect) <= 1e-8)  # allow for near-equality
    assert np.any(np.diff(aspect) < 1e-4)  # require some significant decrease
    # But doesn't decrease by much
    assert np.all(aspect[:-1] / aspect[1:] < 1.1)

    fig = matplotlib.figure.Figure()
    figsize_before = fig.get_figwidth(), fig.get_figheight()
    UpSet(X, element_size=None).make_grid(fig)
    figsize_after = fig.get_figwidth(), fig.get_figheight()
    assert figsize_before == figsize_after

    # TODO: make sure axes are all within figure
    # TODO: make sure text does not overlap axes, even with element_size=None


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
    plot(X, fig, orientation=orientation, show_counts='{:0.2g}')
    assert n_artists_yes_sizes == _count_descendants(fig)
    assert '9.5e+03' in get_all_texts(fig)
    assert '2.8e+02' in get_all_texts(fig)

    fig = matplotlib.figure.Figure()
    plot(X, fig, orientation=orientation, show_percentages=True)
    assert n_artists_yes_sizes == _count_descendants(fig)
    assert '95.5%' in get_all_texts(fig)
    assert '2.8%' in get_all_texts(fig)

    fig = matplotlib.figure.Figure()
    plot(X, fig, orientation=orientation, show_percentages='!{:0.2f}!')
    assert n_artists_yes_sizes == _count_descendants(fig)
    assert '!0.95!' in get_all_texts(fig)
    assert '!0.03!' in get_all_texts(fig)

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


def _get_patch_data(axes, is_vertical):
    out = [{"y": patch.get_y(), "x": patch.get_x(),
            "h": patch.get_height(), "w": patch.get_width(),
            "fc": patch.get_facecolor(),
            "ec": patch.get_edgecolor(),
            "lw": patch.get_linewidth(),
            "ls": patch.get_linestyle(),
            "hatch": patch.get_hatch(),
            }
           for patch in axes.patches]
    if is_vertical:
        out = [{"y": patch["x"], "x": 6.5 - patch["y"],
                "h": patch["w"], "w": patch["h"],
                "fc": patch["fc"],
                "ec": patch["ec"],
                "lw": patch["lw"],
                "ls": patch["ls"],
                "hatch": patch["hatch"],
                }
               for patch in out]
    return pd.DataFrame(out).sort_values("x").reset_index(drop=True)


def _get_color_to_label_from_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    color_to_label = {
        patches[0].get_facecolor(): label
        for patches, label in zip(handles, labels)
    }
    return color_to_label


@pytest.mark.parametrize('orientation', ['horizontal', 'vertical'])
@pytest.mark.parametrize('show_counts', [False, True])
def test_add_stacked_bars(orientation, show_counts):
    df = generate_samples()
    df["label"] = (pd.cut(generate_samples().value + np.random.rand() / 2, 3)
                   .cat.codes
                   .map({0: "foo", 1: "bar", 2: "baz"}))

    upset = UpSet(df, show_counts=show_counts, orientation=orientation)
    upset.add_stacked_bars(by="label")
    upset_axes = upset.plot()

    int_axes = upset_axes["intersections"]
    stacked_axes = upset_axes["extra1"]

    is_vertical = orientation == 'vertical'
    int_rects = _get_patch_data(int_axes, is_vertical)
    stacked_rects = _get_patch_data(stacked_axes, is_vertical)

    # check bar heights match between int_rects and stacked_rects
    assert_series_equal(int_rects.groupby("x")["h"].sum(),
                        stacked_rects.groupby("x")["h"].sum(),
                        check_dtype=False)
    # check count labels match (TODO: check coordinate)
    assert ([elem.get_text() for elem in int_axes.texts] ==
            [elem.get_text() for elem in stacked_axes.texts])

    color_to_label = _get_color_to_label_from_legend(stacked_axes)
    stacked_rects["label"] = stacked_rects["fc"].map(color_to_label)
    # check totals for each label
    assert_series_equal(stacked_rects.groupby("label")["h"].sum(),
                        df.groupby("label").size(),
                        check_dtype=False, check_names=False)

    label_order = [text_obj.get_text()
                   for text_obj in stacked_axes.get_legend().get_texts()]
    # label order should be lexicographic
    assert label_order == sorted(label_order)

    if orientation == "horizontal":
        # order of labels in legend should match stack, top to bottom
        for prev, curr in zip(label_order, label_order[1:]):
            assert (stacked_rects.query("label == @prev")
                    .sort_values("x")["y"].values >=
                    stacked_rects.query("label == @curr")
                    .sort_values("x")["y"].values).all()
    else:
        # order of labels in legend should match stack, left to right
        for prev, curr in zip(label_order, label_order[1:]):
            assert (stacked_rects.query("label == @prev")
                    .sort_values("x")["y"].values <=
                    stacked_rects.query("label == @curr")
                    .sort_values("x")["y"].values).all()


@pytest.mark.parametrize("colors, expected", [
    (["blue", "red", "green"], ["blue", "red", "green"]),
    ({"bar": "blue", "baz": "red", "foo": "green"}, ["blue", "red", "green"]),
    ("Pastel1", ["#fbb4ae", "#b3cde3", "#ccebc5"]),
    (cm.viridis, ["#440154", "#440256", "#450457"]),
    (lambda x: cm.Pastel1(x), ["#fbb4ae", "#b3cde3", "#ccebc5"]),
])
def test_add_stacked_bars_colors(colors, expected):
    df = generate_samples()
    df["label"] = (pd.cut(generate_samples().value + np.random.rand() / 2, 3)
                   .cat.codes
                   .map({0: "foo", 1: "bar", 2: "baz"}))

    upset = UpSet(df)
    upset.add_stacked_bars(by="label", colors=colors,
                           title="Count by gender")
    upset_axes = upset.plot()
    stacked_axes = upset_axes["extra1"]
    color_to_label = _get_color_to_label_from_legend(stacked_axes)
    label_to_color = {v: k for k, v in color_to_label.items()}
    actual = [to_hex(label_to_color[label]) for label in ["bar", "baz", "foo"]]
    expected = [to_hex(color) for color in expected]
    assert actual == expected


@pytest.mark.parametrize('int_sum_over', [False, True])
@pytest.mark.parametrize('stack_sum_over', [False, True])
@pytest.mark.parametrize('show_counts', [False, True])
def test_add_stacked_bars_sum_over(int_sum_over, stack_sum_over, show_counts):
    # A rough test of sum_over
    df = generate_samples()
    df["label"] = (pd.cut(generate_samples().value + np.random.rand() / 2, 3)
                   .cat.codes
                   .map({0: "foo", 1: "bar", 2: "baz"}))

    upset = UpSet(df, sum_over="value" if int_sum_over else None,
                  show_counts=show_counts)
    upset.add_stacked_bars(by="label",
                           sum_over="value" if stack_sum_over else None,
                           colors='Pastel1')
    upset_axes = upset.plot()

    int_axes = upset_axes["intersections"]
    stacked_axes = upset_axes["extra1"]

    int_rects = _get_patch_data(int_axes, is_vertical=False)
    stacked_rects = _get_patch_data(stacked_axes, is_vertical=False)

    if int_sum_over == stack_sum_over:
        # check bar heights match between int_rects and stacked_rects
        assert_series_equal(int_rects.groupby("x")["h"].sum(),
                            stacked_rects.groupby("x")["h"].sum(),
                            check_dtype=False)
        # and check labels match with show_counts
        assert ([elem.get_text() for elem in int_axes.texts] ==
                [elem.get_text() for elem in stacked_axes.texts])
    else:
        assert (int_rects.groupby("x")["h"].sum() !=
                stacked_rects.groupby("x")["h"].sum()).all()
        if show_counts:
            assert ([elem.get_text() for elem in int_axes.texts] !=
                    [elem.get_text() for elem in stacked_axes.texts])


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


@pytest.mark.parametrize(
    "filter_params, expected",
    [
        ({"min_subset_size": 623},
         {(True, False, False): 884,
          (True, True, False): 1547,
          (True, False, True): 623,
          (True, True, True): 990,
          }),
        ({"min_subset_size": 800, "max_subset_size": 990},
         {(True, False, False): 884,
          (True, True, True): 990,
          }),
        ({"min_degree": 2},
         {(True, True, False): 1547,
          (True, False, True): 623,
          (False, True, True): 258,
          (True, True, True): 990,
          }),
        ({"min_degree": 2, "max_degree": 2},
         {(True, True, False): 1547,
          (True, False, True): 623,
          (False, True, True): 258,
          }),
        ({"max_subset_size": 500, "max_degree": 2},
         {(False, False, False): 220,
          (False, True, False): 335,
          (False, False, True): 143,
          (False, True, True): 258,
          }),
    ]
)
@pytest.mark.parametrize('sort_by', ['cardinality', 'degree'])
def test_filter_subsets(filter_params, expected, sort_by):
    data = generate_samples(seed=0, n_samples=5000, n_categories=3)
    # data =
    #   cat1   cat0   cat2
    #   False  False  False     220
    #   True   False  False     884
    #   False  True   False     335
    #          False  True      143
    #   True   True   False    1547
    #          False  True      623
    #   False  True   True      258
    #   True   True   True      990
    upset_full = UpSet(data, subset_size='auto', sort_by=sort_by)
    upset_filtered = UpSet(data, subset_size='auto',
                           sort_by=sort_by,
                           **filter_params)
    intersections = upset_full.intersections
    df = upset_full._df
    # check integrity of expected, just to be sure
    for key, value in expected.items():
        assert intersections.loc[key] == value
    subset_intersections = intersections[
        intersections.index.isin(list(expected.keys()))]
    subset_df = df[df.index.isin(list(expected.keys()))]
    assert len(subset_intersections) < len(intersections)
    assert_series_equal(upset_filtered.intersections, subset_intersections)
    assert_frame_equal(upset_filtered._df.drop("_bin", axis=1),
                       subset_df.drop("_bin", axis=1))
    # category totals should not be affected
    assert_series_equal(upset_full.totals, upset_filtered.totals)


@pytest.mark.parametrize('x', [
    generate_counts(n_categories=3),
    generate_counts(n_categories=8),
    generate_counts(n_categories=15),
])
@pytest.mark.parametrize('orientation', [
    'horizontal',
    'vertical',
])
def test_matrix_plot_margins(x, orientation):
    """Non-regression test addressing a bug where there is are large whitespace
       margins around the matrix when the number of intersections is large"""
    axes = plot(x, orientation=orientation)

    # Expected behavior is that each matrix column takes up one unit on x-axis
    expected_width = len(x)
    attr = 'get_xlim' if orientation == 'horizontal' else 'get_ylim'
    lim = getattr(axes['matrix'], attr)()
    assert expected_width == lim[1] - lim[0]


def _make_facecolor_list(colors):
    return [{"facecolor": c} for c in colors]


CAT1_2_RED_STYLES = _make_facecolor_list(["blue", "blue", "blue", "blue",
                                          "red", "blue", "blue", "red"])
CAT1_RED_STYLES = _make_facecolor_list(["blue", "red", "blue", "blue",
                                        "red", "red", "blue", "red"])
CAT_NOT1_RED_STYLES = _make_facecolor_list(["red", "blue", "red", "red",
                                            "blue", "blue", "red", "blue"])
CAT1_NOT2_RED_STYLES = _make_facecolor_list(["blue", "red", "blue", "blue",
                                             "blue", "red", "blue", "blue"])
CAT_NOT1_2_RED_STYLES = _make_facecolor_list(["red", "blue", "blue", "red",
                                              "blue", "blue", "blue", "blue"])


@pytest.mark.parametrize(
    "kwarg_list,expected_subset_styles,expected_legend",
    [
        # Different forms of including two categories
        ([{"present": ["cat1", "cat2"], "facecolor": "red"}],
         CAT1_2_RED_STYLES, []),
        ([{"present": {"cat1", "cat2"}, "facecolor": "red"}],
         CAT1_2_RED_STYLES, []),
        ([{"present": ("cat1", "cat2"), "facecolor": "red"}],
         CAT1_2_RED_STYLES, []),
        # with legend
        ([{"present": ("cat1", "cat2"), "facecolor": "red", "label": "foo"}],
         CAT1_2_RED_STYLES, [({"facecolor": "red"}, "foo")]),
        # present only cat1
        ([{"present": ("cat1",), "facecolor": "red"}],
         CAT1_RED_STYLES, []),
        ([{"present": "cat1", "facecolor": "red"}],
         CAT1_RED_STYLES, []),
        # Some uses of absent
        ([{"absent": "cat1", "facecolor": "red"}],
         CAT_NOT1_RED_STYLES, []),
        ([{"present": "cat1", "absent": ["cat2"], "facecolor": "red"}],
         CAT1_NOT2_RED_STYLES, []),
        ([{"absent": ["cat2", "cat1"], "facecolor": "red"}],
         CAT_NOT1_2_RED_STYLES, []),
        # min/max args
        ([{"present": ["cat1", "cat2"], "min_degree": 3, "facecolor": "red"}],
         _make_facecolor_list(["blue"] * 7 + ["red"]), []),
        ([{"present": ["cat1", "cat2"], "max_subset_size": 3000,
           "facecolor": "red"}],
         _make_facecolor_list(["blue"] * 7 + ["red"]), []),
        ([{"present": ["cat1", "cat2"], "max_degree": 2, "facecolor": "red"}],
         _make_facecolor_list(["blue"] * 4 + ["red"] + ["blue"] * 3), []),
        ([{"present": ["cat1", "cat2"], "min_subset_size": 3000,
           "facecolor": "red"}],
         _make_facecolor_list(["blue"] * 4 + ["red"] + ["blue"] * 3), []),
        # cat1 _or_ cat2
        ([{"present": "cat1", "facecolor": "red"},
          {"present": "cat2", "facecolor": "red"}],
         _make_facecolor_list(["blue", "red", "red", "blue",
                               "red", "red", "red", "red"]), []),
        # With multiple uses of label
        ([{"present": "cat1", "facecolor": "red", "label": "foo"},
          {"present": "cat2", "facecolor": "red", "label": "bar"}],
         _make_facecolor_list(["blue", "red", "red", "blue",
                               "red", "red", "red", "red"]),
         [({"facecolor": "red"}, "foo; bar")]),
        ([{"present": "cat1", "facecolor": "red", "label": "foo"},
          {"present": "cat2", "facecolor": "red", "label": "foo"}],
         _make_facecolor_list(["blue", "red", "red", "blue",
                               "red", "red", "red", "red"]),
         [({"facecolor": "red"}, "foo")]),
        # With multiple colours, the latest overrides
        ([{"present": "cat1", "facecolor": "red", "label": "foo"},
          {"present": "cat2", "facecolor": "green", "label": "bar"}],
         _make_facecolor_list(["blue", "red", "green", "blue",
                               "green", "red", "green", "green"]),
         [({"facecolor": "red"}, "foo"),
          ({"facecolor": "green"}, "bar")]),
        # Combining multiple style properties
        ([{"present": "cat1", "facecolor": "red", "hatch": "//"},
          {"present": "cat2", "edgecolor": "green", "linestyle": "dotted"}],
         [{"facecolor": "blue"},
          {"facecolor": "red", "hatch": "//"},
          {"facecolor": "blue", "edgecolor": "green", "linestyle": "dotted"},
          {"facecolor": "blue"},
          {"facecolor": "red", "hatch": "//", "edgecolor": "green",
           "linestyle": "dotted"},
          {"facecolor": "red", "hatch": "//"},
          {"facecolor": "blue", "edgecolor": "green",
           "linestyle": "dotted"},
          {"facecolor": "red", "hatch": "//", "edgecolor": "green",
           "linestyle": "dotted"},
          ],
         []),
    ])
def test_style_subsets(kwarg_list, expected_subset_styles, expected_legend):
    data = generate_counts()
    upset = UpSet(data, facecolor="blue")
    for kw in kwarg_list:
        upset.style_subsets(**kw)
    actual_subset_styles = upset.subset_styles
    assert actual_subset_styles == expected_subset_styles
    assert upset.subset_legend == expected_legend


def _dots_to_dataframe(ax, is_vertical):
    matrix_path_collection = ax.collections[0]
    matrix_dots = pd.DataFrame(
        matrix_path_collection.get_offsets(), columns=["x", "y"]
    ).join(
        pd.DataFrame(matrix_path_collection.get_facecolors(),
                     columns=["fc_r", "fc_g", "fc_b", "fc_a"]),
    ).join(
        pd.DataFrame(matrix_path_collection.get_edgecolors(),
                     columns=["ec_r", "ec_g", "ec_b", "ec_a"]),
    ).assign(
        lw=matrix_path_collection.get_linewidths(),
        ls=matrix_path_collection.get_linestyles(),
        hatch=matrix_path_collection.get_hatch(),
    )

    matrix_dots["ls_offset"] = matrix_dots["ls"].map(
        lambda tup: tup[0]).astype(float)
    matrix_dots["ls_seq"] = matrix_dots["ls"].map(
        lambda tup: None if tup[1] is None else tuple(tup[1]))
    del matrix_dots["ls"]

    if is_vertical:
        matrix_dots[["x", "y"]] = matrix_dots[["y", "x"]]
        matrix_dots["x"] = 7 - matrix_dots["x"]
    return matrix_dots


@pytest.mark.parametrize('orientation', ['horizontal', 'vertical'])
def test_style_subsets_artists(orientation):
    # Check that subset_styles are all appropriately reflected in matplotlib
    # artists.
    # This may be a bit overkill, and too coupled with implementation details.
    is_vertical = orientation == 'vertical'
    data = generate_counts()
    upset = UpSet(data, orientation=orientation)
    subset_styles = [
        {"facecolor": "black"},
        {"facecolor": "red"},
        {"edgecolor": "red"},
        {"edgecolor": "red", "linewidth": 4},
        {"linestyle": "dotted"},
        {"edgecolor": "red", "facecolor": "blue", "hatch": "//"},
        {"facecolor": "blue"},
        {},
    ]

    if is_vertical:
        upset.subset_styles = subset_styles[::-1]
    else:
        upset.subset_styles = subset_styles

    upset_axes = upset.plot()

    int_rects = _get_patch_data(upset_axes["intersections"], is_vertical)
    int_rects[["fc_r", "fc_g", "fc_b", "fc_a"]] = (
        int_rects.pop("fc").apply(lambda x: pd.Series(x)))
    int_rects[["ec_r", "ec_g", "ec_b", "ec_a"]] = (
        int_rects.pop("ec").apply(lambda x: pd.Series(x)))
    int_rects["ls_is_solid"] = int_rects.pop("ls").map(
        lambda x: x == "solid" or pd.isna(x))
    expected = pd.DataFrame({
        "fc_r": [0, 1, 0, 0, 0, 0, 0, 0],
        "fc_g": [0, 0, 0, 0, 0, 0, 0, 0],
        "fc_b": [0, 0, 0, 0, 0, 1, 1, 0],
        "ec_r": [0, 1, 1, 1, 0, 1, 0, 0],
        "ec_g": [0, 0, 0, 0, 0, 0, 0, 0],
        "ec_b": [0, 0, 0, 0, 0, 0, 1, 0],
        "lw": [1, 1, 1, 4, 1, 1, 1, 1],
        "ls_is_solid": [True, True, True, True, False, True, True, True],
    })

    assert_frame_equal(expected, int_rects[expected.columns],
                       check_dtype=False)

    styled_dots = _dots_to_dataframe(upset_axes["matrix"], is_vertical)
    baseline_dots = _dots_to_dataframe(
        UpSet(data, orientation=orientation).plot()["matrix"],
        is_vertical
    )
    inactive_dot_mask = (baseline_dots[["fc_a"]] < 1).values.ravel()
    assert_frame_equal(baseline_dots.loc[inactive_dot_mask],
                       styled_dots.loc[inactive_dot_mask])

    styled_dots = styled_dots.loc[~inactive_dot_mask]

    styled_dots = styled_dots.drop(columns="y").groupby("x").apply(
        lambda df: df.drop_duplicates())
    styled_dots["ls_is_solid"] = styled_dots.pop("ls_seq").isna()
    assert_frame_equal(expected.iloc[1:].reset_index(drop=True),
                       styled_dots[expected.columns].reset_index(drop=True),
                       check_dtype=False)

    # TODO: check lines between dots
    # matrix_line_collection = upset_axes["matrix"].collections[1]


def test_many_categories():
    # Tests regressions against GH#193
    n_cats = 250
    index1 = [True, False] + [False] * (n_cats - 2)
    index2 = [False, True] + [False] * (n_cats - 2)
    columns = [chr(i + 33) for i in range(n_cats)]
    data = pd.DataFrame([index1, index2], columns=columns)
    data["value"] = 1
    data = data.set_index(columns)["value"]
    UpSet(data)
