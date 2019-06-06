from __future__ import print_function, division, absolute_import

import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.tight_layout import get_renderer


def _aggregate_data(df, subset_size, sum_over):
    """
    Returns
    -------
    df : DataFrame
        full data frame
    aggregated : Series
        aggregates
    """
    _SUBSET_SIZE_VALUES = ['auto', 'count', 'sum', 'legacy']
    if subset_size not in _SUBSET_SIZE_VALUES:
        raise ValueError('subset_size should be one of %s. Got %r'
                         % (_SUBSET_SIZE_VALUES, subset_size))
    if df.ndim == 1:
        # Series
        input_name = df.name
        df = pd.DataFrame({'_value': df})

        if not df.index.is_unique:
            if subset_size == 'legacy':
                warnings.warn('From version 0.4, passing a Series as data '
                              'with non-unqiue groups will raise an error '
                              'unless subset_size="sum" or "count".',
                              FutureWarning)
            if subset_size == 'auto':
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
        if subset_size == 'legacy' and sum_over is None:
            raise ValueError('Please specify subset_size or sum_over for a '
                             'DataFrame.')
        elif subset_size == 'legacy' and sum_over is False:
            warnings.warn('sum_over=False will not be supported from version '
                          '0.4. Use subset_size="auto" or "count" '
                          'instead.', DeprecationWarning)
        elif subset_size in ('auto', 'sum') and sum_over is False:
            # remove this after deprecation
            raise ValueError('sum_over=False is not supported when '
                             'subset_size=%r' % subset_size)
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

    gb = df.groupby(level=list(range(df.index.nlevels)))
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


def _process_data(df, sort_by, sort_categories_by, subset_size, sum_over):
    df, agg = _aggregate_data(df, subset_size, sum_over)

    # check all indices are boolean
    assert all(set([True, False]) >= set(level) for level in agg.index.levels)

    totals = [agg[agg.index.get_level_values(name).values.astype(bool)].sum()
              for name in agg.index.names]
    totals = pd.Series(totals, index=agg.index.names)
    if sort_categories_by == 'cardinality':
        totals.sort_values(ascending=False, inplace=True)
    elif sort_categories_by is not None:
        raise ValueError('Unknown sort_categories_by: %r' % sort_categories_by)
    df = df.reorder_levels(totals.index.values)
    agg = agg.reorder_levels(totals.index.values)

    if sort_by == 'cardinality':
        agg = agg.sort_values(ascending=False)
    elif sort_by == 'degree':
        comb = itertools.combinations
        o = pd.DataFrame([{name: True for name in names}
                          for i in range(agg.index.nlevels + 1)
                          for names in comb(agg.index.names, i)],
                         columns=agg.index.names)
        o.fillna(False, inplace=True)
        o = o.astype(bool)
        o.set_index(agg.index.names, inplace=True)
        agg = agg.reindex(index=o.index)
    else:
        raise ValueError('Unknown sort_by: %r' % sort_by)

    min_value = 0
    max_value = np.inf
    agg = agg[np.logical_and(agg >= min_value, agg <= max_value)]

    # add '_bin' to df indicating index in agg
    # XXX: ugly!
    def _pack_binary(X):
        X = pd.DataFrame(X)
        out = 0
        for i, (_, col) in enumerate(X.items()):
            out *= 2
            out += col
        return out

    df_packed = _pack_binary(df.index.to_frame())
    data_packed = _pack_binary(agg.index.to_frame())
    df['_bin'] = pd.Series(df_packed).map(
        pd.Series(np.arange(len(data_packed)),
                  index=data_packed))

    return df, agg, totals


class _Transposed:
    """Wrap an object in order to transpose some plotting operations

    Attributes of obj will be mapped.
    Keyword arguments when calling obj will be mapped.

    The mapping is not recursive: callable attributes need to be _Transposed
    again.
    """

    def __init__(self, obj):
        self.__obj = obj

    def __getattr__(self, key):
        return getattr(self.__obj, self._NAME_TRANSPOSE.get(key, key))

    def __call__(self, *args, **kwargs):
        return self.__obj(*args, **{self._NAME_TRANSPOSE.get(k, k): v
                                    for k, v in kwargs.items()})

    _NAME_TRANSPOSE = {
        'width': 'height',
        'height': 'width',
        'hspace': 'wspace',
        'wspace': 'hspace',
        'hlines': 'vlines',
        'vlines': 'hlines',
        'bar': 'barh',
        'barh': 'bar',
        'xaxis': 'yaxis',
        'yaxis': 'xaxis',
        'left': 'bottom',
        'right': 'top',
        'top': 'right',
        'bottom': 'left',
        'sharex': 'sharey',
        'sharey': 'sharex',
        'get_figwidth': 'get_figheight',
        'get_figheight': 'get_figwidth',
        'set_figwidth': 'set_figheight',
        'set_figheight': 'set_figwidth',
        'set_xlabel': 'set_ylabel',
        'set_ylabel': 'set_xlabel',
    }


def _transpose(obj):
    if isinstance(obj, str):
        return _Transposed._NAME_TRANSPOSE.get(obj, obj)
    return _Transposed(obj)


def _identity(obj):
    return obj


class UpSet:
    """Manage the data and drawing for a basic UpSet plot

    Primary public method is :meth:`plot`.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Elements associated with categories (a DataFrame), or the size of each
        subset of categories (a Series).
        Should have MultiIndex where each level is binary,
        corresponding to category membership.
        If a DataFrame, `sum_over` must be a string or False.
    orientation : {'horizontal' (default), 'vertical'}
        If horizontal, intersections are listed from left to right.
    sort_by : {'cardinality', 'degree'}
        If 'cardinality', subset are listed from largest to smallest.
        If 'degree', they are listed in order of the number of categories
        intersected.
    sort_categories_by : {'cardinality', None}
        Whether to sort the categories by total cardinality, or leave them
        in the provided order.

        .. versionadded: 0.3
            Replaces sort_sets_by
    subset_size : {'auto', 'count', 'sum'}
        Configures how to calculate the size of a subset. Choices are:

        'auto'
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

        Until version 0.4, the default is 'legacy' which uses `sum_over` to
        control this behaviour. From version 0.4, 'auto' will be default.
    sum_over : str or None
        If `subset_size='sum'` or `'auto'`, then the intersection size is the
        sum of the specified field in the `data` DataFrame. If a Series, only
        None is supported and its value is summed.

        If `subset_size='legacy'`, `sum_over` must be specified when `data` is
        a DataFrame. If False, the intersection plot will show the count of
        each subset. Otherwise, it shows the sum of the specified field.
    facecolor : str
        Color for bar charts and dots.
    with_lines : bool
        Whether to show lines joining dots in the matrix, to mark multiple
        categories being intersected.
    element_size : float or None
        Side length in pt. If None, size is estimated to fit figure
    intersection_plot_elements : int
        The intersections plot should be large enough to fit this many matrix
        elements.
    totals_plot_elements : int
        The totals plot should be large enough to fit this many matrix
        elements.
    show_counts : bool or str, default=False
        Whether to label the intersection size bars with the cardinality
        of the intersection. When a string, this formats the number.
        For example, '%d' is equivalent to True.
    sort_sets_by
        .. deprecated: 0.3
            Replaced by sort_categories_by, this parameter will be removed in
            version 0.4.
    """
    _default_figsize = (10, 6)

    def __init__(self, data, orientation='horizontal', sort_by='degree',
                 sort_categories_by='cardinality',
                 subset_size='legacy', sum_over=None,
                 facecolor='black',
                 with_lines=True, element_size=32,
                 intersection_plot_elements=6, totals_plot_elements=2,
                 show_counts='', sort_sets_by='deprecated'):

        self._horizontal = orientation == 'horizontal'
        self._reorient = _identity if self._horizontal else _transpose
        self._facecolor = facecolor
        self._with_lines = with_lines
        self._element_size = element_size
        self._totals_plot_elements = totals_plot_elements
        self._subset_plots = [{'type': 'default',
                               'id': 'intersections',
                               'elements': intersection_plot_elements}]
        self._show_counts = show_counts

        if sort_sets_by != 'deprecated':
            sort_categories_by = sort_sets_by
            warnings.warn('sort_sets_by was deprecated in version 0.3 and '
                          'will be removed in version 0.4', DeprecationWarning)

        (self._df, self.intersections,
         self.totals) = _process_data(data,
                                      sort_by=sort_by,
                                      sort_categories_by=sort_categories_by,
                                      subset_size=subset_size,
                                      sum_over=sum_over)
        if not self._horizontal:
            self.intersections = self.intersections[::-1]

    def _swapaxes(self, x, y):
        if self._horizontal:
            return x, y
        return y, x

    def add_catplot(self, kind, value=None, elements=3, **kw):
        """Add a seaborn catplot over subsets when :func:`plot` is called.

        Parameters
        ----------
        kind : str
            One of {"point", "bar", "strip", "swarm", "box", "violin", "boxen"}
        value : str, optional
            Column name for the value to plot (i.e. y if
            orientation='horizontal'), required if `data` is a DataFrame.
        elements : int, default=3
            Size of the axes counted in number of matrix elements.
        **kw : dict
            Additional keywords to pass to :func:`seaborn.catplot`.

            Our implementation automatically determines 'ax', 'data', 'x', 'y'
            and 'orient', so these are prohibited keys in `kw`.

        Returns
        -------
        None
        """
        assert not set(kw.keys()) & {'ax', 'data', 'x', 'y', 'orient'}
        if value is None:
            if '_value' not in self._df.columns:
                raise ValueError('value cannot be set if data is a Series. '
                                 'Got %r' % value)
        else:
            if value not in self._df.columns:
                raise ValueError('value %r is not a column in data' % value)
        self._subset_plots.append({'type': 'catplot',
                                   'value': value,
                                   'kind': kind,
                                   'id': 'extra%d' % len(self._subset_plots),
                                   'elements': elements,
                                   'kw': kw})

    def _plot_catplot(self, ax, value, kind, kw):
        df = self._df
        if value is None and '_value' in df.columns:
            value = '_value'
        elif value is None:
            raise ValueError('value can only be None when data is a Series')
        kw = kw.copy()
        if self._horizontal:
            kw['orient'] = 'v'
            kw['x'] = '_bin'
            kw['y'] = value
        else:
            kw['orient'] = 'h'
            kw['x'] = value
            kw['y'] = '_bin'
        import seaborn
        kw['ax'] = ax
        getattr(seaborn, kind + 'plot')(data=df, **kw)

        ax = self._reorient(ax)
        if value == '_value':
            ax.set_ylabel('')

        ax.xaxis.set_visible(False)
        for x in ['top', 'bottom', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)

        tick_axis = ax.yaxis
        tick_axis.grid(True)

    def make_grid(self, fig=None):
        """Get a SubplotSpec for each Axes, accounting for label text width
        """
        n_cats = len(self.totals)
        n_inters = len(self.intersections)

        if fig is None:
            fig = plt.gcf()

        # Determine text size to determine figure size / spacing
        r = get_renderer(fig)
        t = fig.text(0, 0, '\n'.join(self.totals.index.values))
        textw = t.get_window_extent(renderer=r).width
        t.remove()

        MAGIC_MARGIN = 10  # FIXME
        figw = self._reorient(fig.get_window_extent(renderer=r)).width

        sizes = np.asarray([p['elements'] for p in self._subset_plots])

        if self._element_size is None:
            colw = (figw - textw - MAGIC_MARGIN) / (len(self.intersections) +
                                                    self._totals_plot_elements)
        else:
            fig = self._reorient(fig)
            render_ratio = figw / fig.get_figwidth()
            colw = self._element_size / 72 * render_ratio
            figw = (colw * (len(self.intersections) +
                            self._totals_plot_elements) +
                    MAGIC_MARGIN + textw)
            fig.set_figwidth(figw / render_ratio)
            fig.set_figheight((colw * (n_cats + sizes.sum())) /
                              render_ratio)

        text_nelems = int(np.ceil(figw / colw - (len(self.intersections) +
                                                 self._totals_plot_elements)))

        GS = self._reorient(matplotlib.gridspec.GridSpec)
        gridspec = GS(*self._swapaxes(n_cats + sizes.sum(),
                                      n_inters + text_nelems +
                                      self._totals_plot_elements),
                      hspace=1)
        if self._horizontal:
            out = {'matrix': gridspec[-n_cats:, -n_inters:],
                   'shading': gridspec[-n_cats:, :],
                   'totals': gridspec[-n_cats:, :self._totals_plot_elements],
                   'gs': gridspec}
            cumsizes = np.cumsum(sizes[::-1])
            for start, stop, plot in zip(np.hstack([[0], cumsizes]), cumsizes,
                                         self._subset_plots[::-1]):
                out[plot['id']] = gridspec[start:stop, -n_inters:]
        else:
            out = {'matrix': gridspec[-n_inters:, :n_cats],
                   'shading': gridspec[:, :n_cats],
                   'totals': gridspec[:self._totals_plot_elements, :n_cats],
                   'gs': gridspec}
            cumsizes = np.cumsum(sizes)
            for start, stop, plot in zip(np.hstack([[0], cumsizes]), cumsizes,
                                         self._subset_plots):
                out[plot['id']] = gridspec[-n_inters:,
                                           start + n_cats:stop + n_cats]
        return out

    def plot_matrix(self, ax):
        """Plot the matrix of intersection indicators onto ax
        """
        ax = self._reorient(ax)
        data = self.intersections
        n_cats = data.index.nlevels

        idx = np.flatnonzero(data.index.to_frame()[data.index.names].values)
        c = np.array(['lightgrey'] * len(data) * n_cats, dtype='O')
        c[idx] = self._facecolor
        x = np.repeat(np.arange(len(data)), n_cats)
        y = np.tile(np.arange(n_cats), len(data))
        if self._element_size is not None:
            s = (self._element_size * .35) ** 2
        else:
            # TODO: make s relative to colw
            s = 200
        ax.scatter(*self._swapaxes(x, y), c=c.tolist(), linewidth=0, s=s)

        if self._with_lines:
            line_data = (pd.Series(y[idx], index=x[idx])
                         .groupby(level=0)
                         .aggregate(['min', 'max']))
            ax.vlines(line_data.index.values,
                      line_data['min'], line_data['max'],
                      lw=2, colors=self._facecolor)

        tick_axis = ax.yaxis
        tick_axis.set_ticks(np.arange(n_cats))
        tick_axis.set_ticklabels(data.index.names,
                                 rotation=0 if self._horizontal else -90)
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        if not self._horizontal:
            ax.yaxis.set_ticks_position('top')
        ax.set_frame_on(False)

    def plot_intersections(self, ax):
        """Plot bars indicating intersection size
        """
        ax = self._reorient(ax)
        rects = ax.bar(np.arange(len(self.intersections)), self.intersections,
                       .5, color=self._facecolor, zorder=10, align='center')

        self._label_sizes(ax, rects, 'top' if self._horizontal else 'right')

        ax.xaxis.set_visible(False)
        for x in ['top', 'bottom', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)

        tick_axis = ax.yaxis
        tick_axis.grid(True)
        ax.set_ylabel('Intersection size')

    def _label_sizes(self, ax, rects, where):
        if not self._show_counts:
            return
        fmt = '%d' if self._show_counts is True else self._show_counts
        if where == 'right':
            margin = 0.01 * abs(np.diff(ax.get_xlim()))
            for rect in rects:
                width = rect.get_width()
                ax.text(width + margin,
                        rect.get_y() + rect.get_height() * .5,
                        fmt % width,
                        ha='left', va='center')
        elif where == 'left':
            margin = 0.01 * abs(np.diff(ax.get_xlim()))
            for rect in rects:
                width = rect.get_width()
                ax.text(width + margin,
                        rect.get_y() + rect.get_height() * .5,
                        fmt % width,
                        ha='right', va='center')
        elif where == 'top':
            margin = 0.01 * abs(np.diff(ax.get_ylim()))
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() * .5,
                        height + margin, fmt % height,
                        ha='center', va='bottom')
        else:
            raise NotImplementedError('unhandled where: %r' % where)

    def plot_totals(self, ax):
        """Plot bars indicating total set size
        """
        orig_ax = ax
        ax = self._reorient(ax)
        rects = ax.barh(np.arange(len(self.totals.index.values)), self.totals,
                        .5, color=self._facecolor, align='center')
        self._label_sizes(ax, rects, 'left' if self._horizontal else 'top')

        max_total = self.totals.max()
        if self._horizontal:
            orig_ax.set_xlim(max_total, 0)
        for x in ['top', 'left', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.grid(True)
        ax.patch.set_visible(False)

    def plot_shading(self, ax):
        # alternating row shading (XXX: use add_patch(Rectangle)?)
        for i in range(0, len(self.totals), 2):
            rect = plt.Rectangle(self._swapaxes(0, i - .4),
                                 *self._swapaxes(*(1, .8)),
                                 facecolor='#f5f5f5', lw=0, zorder=0)
            ax.add_patch(rect)
        ax.set_frame_on(False)
        ax.tick_params(
            axis='both',
            which='both',
            left=False,
            right=False,
            bottom=False,
            top=False,
            labelbottom=False,
            labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot(self, fig=None):
        """Draw all parts of the plot onto fig or a new figure

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Defaults to a new figure.

        Returns
        -------
        subplots : dict of matplotlib.axes.Axes
            Keys are 'matrix', 'intersections', 'totals', 'shading'
        """
        if fig is None:
            fig = plt.figure(figsize=self._default_figsize)
        specs = self.make_grid(fig)
        shading_ax = fig.add_subplot(specs['shading'])
        self.plot_shading(shading_ax)
        matrix_ax = self._reorient(fig.add_subplot)(specs['matrix'],
                                                    sharey=shading_ax)
        self.plot_matrix(matrix_ax)
        totals_ax = self._reorient(fig.add_subplot)(specs['totals'],
                                                    sharey=matrix_ax)
        self.plot_totals(totals_ax)
        out = {'matrix': matrix_ax,
               'shading': shading_ax,
               'totals': totals_ax}

        for plot in self._subset_plots:
            ax = self._reorient(fig.add_subplot)(specs[plot['id']],
                                                 sharex=matrix_ax)
            if plot['type'] == 'default':
                self.plot_intersections(ax)
            elif plot['type'] == 'catplot':
                self._plot_catplot(ax, plot['value'], plot['kind'], plot['kw'])
            else:
                raise ValueError('Unknown subset plot type: %r' % plot['type'])
            out[plot['id']] = ax
        return out

    def _repr_html_(self):
        fig = plt.figure(figsize=self._default_figsize)
        self.plot(fig=fig)
        return fig._repr_html_()


def plot(data, fig=None, **kwargs):
    """Make an UpSet plot of data on fig

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Values for each set to plot.
        Should have multi-index where each level is binary,
        corresponding to set membership.
        If a DataFrame, `sum_over` must be a string or False.
    fig : matplotlib.figure.Figure, optional
        Defaults to a new figure.
    kwargs
        Other arguments for :class:`UpSet`

    Returns
    -------
    subplots : dict of matplotlib.axes.Axes
        Keys are 'matrix', 'intersections', 'totals', 'shading'
    """
    return UpSet(data, **kwargs).plot(fig)
