import itertools

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.tight_layout import get_renderer


def _process_data(data, order, order_categories):
    # check all indices are vertical
    assert all(set([True, False]) >= set(level) for level in data.index.levels)
    if not data.index.is_unique:
        data = (data
                .groupby(level=list(range(data.index.nlevels)))
                .sum())

    totals = []
    for i in range(data.index.nlevels):
        idxslice = pd.IndexSlice[(slice(None),) * i + (True,)]
        # FIXME: can get IndexingError if level only contains False
        totals.append(data.loc[idxslice].sum())
    totals = pd.Series(totals, index=data.index.names)
    if order_categories:
        totals.sort_values(ascending=False, inplace=True)
    data = data.reorder_levels(totals.index.values)

    if order == 'size':
        data = data.sort_values(ascending=False)
    elif order == 'degree':
        comb = itertools.combinations
        o = pd.DataFrame([{name: True for name in names}
                          for i in range(data.index.nlevels)
                          for names in comb(data.index.names, i)],
                         columns=data.index.names)
        o.fillna(False, inplace=True)
        o = o.astype(bool)
        o.set_index(data.index.names, inplace=True)
        # FIXME: should use reindex(index=...) ??
        data = data.loc[o.index]
    else:
        raise ValueError('Unknown order: %r' % order)

    min_value = 0
    max_value = np.inf
    data = data[np.logical_and(data >= min_value, data <= max_value)]

    return data, totals


class UpSet:
    """Manage the data and drawing for a basic UpSet plot

    Primary public method is :meth:`plot`.

    Parameters
    ----------
    data : pandas.Series
        Values for each set to plot.
        Should have multi-index where each level is binary,
        corresponding to set membership.
    vert : bool
        If True, the primary plot (bar chart of intersections) will
        be vertical.
    order : {'size', 'degree'}
        If 'size', set intersections are listed from largest to smallest value.
        If 'degree', they are listed in order of the number of sets
        intersected.
    order_categories : bool
        Whether to order the categories by total value, or leave them
        in the provided order.
    forecolor : str
        Color for bar charts and dots.
    with_lines : bool
        Whether to show lines joining dots in the matrix, to mark multiple sets
        being intersected.
    intersections_plot_size : float
        The intersections plot should be large enough to fit this many matrix
        dots.
    totals_plot_size : float
        The totals plot should be large enough to fit this many matrix
        dots.
    """

    def __init__(self, data, vert=True, order='degree',
                 order_categories=True, forecolor='black',
                 with_lines=True, intersection_plot_size=6,
                 totals_plot_size=5):

        self._vert = vert
        if not vert:
            raise NotImplementedError()
        self._forecolor = forecolor
        self._with_lines = with_lines
        self._totals_plot_size = totals_plot_size
        self._intersection_plot_size = intersection_plot_size

        (self.intersections,
         self.totals) = _process_data(data,
                                      order=order,
                                      order_categories=order_categories)

    def make_grid(self, fig=None):
        """Get a SubplotSpec for each Axes, accounting for label text width
        """
        n_cats = len(self.totals)
        n_inters = len(self.intersections)

        text_space = self._calculate_text_ncols(fig)
        GS = matplotlib.gridspec.GridSpec
        gridspec = GS(n_cats + self._intersection_plot_size,
                      n_inters + text_space + self._totals_plot_size,
                      hspace=1)
        return {'intersections': gridspec[:-n_cats, -n_inters:],
                'matrix': gridspec[-n_cats:, -n_inters:],
                'totals': gridspec[-n_cats:, :self._totals_plot_size],
                'gs': gridspec}

    def _calculate_text_ncols(self, fig):
        if fig is None:
            fig = plt.gcf()
        r = get_renderer(fig)
        t = fig.text(0, 0, '\n'.join(self.totals.index.values))
        textw = t.get_window_extent(renderer=r).width
        figw = fig.get_window_extent(renderer=r).width
        MAGIC_MARGIN = 20  # FIXME
        colw = (figw - textw - MAGIC_MARGIN) / (len(self.intersections) +
                                                self._totals_plot_size)
        t.remove()
        return int(np.ceil(figw / colw - (len(self.intersections) +
                                          self._totals_plot_size)))

    def plot_matrix(self, ax):
        """Plot the matrix of intersection indicators onto ax
        """
        data = self.intersections
        n_categories = data.index.nlevels

        # alternating row shading (XXX: use add_patch(Rectangle)?)
        alternating = np.arange(0, n_categories, 2)
        ax.barh(alternating, np.full(len(alternating), len(data) + 1),
                left=-1, color='#f5f5f5', zorder=0, linewidth=0,
                align='center')

        idx = np.flatnonzero(data.index.to_frame()[data.index.names].values)
        c = np.array(['lightgrey'] * len(data) * n_categories, dtype='O')
        c[idx] = self._forecolor
        x = np.repeat(np.arange(len(data)), n_categories)
        y = np.tile(np.arange(n_categories), len(data))
        # TODO: make s relative to colw
        ax.scatter(x, y, c=c.tolist(), linewidth=0, s=200)

        if self._with_lines:
            line_data = (pd.Series(y[idx], index=x[idx])
                         .groupby(level=0)
                         .aggregate(['min', 'max']))
            ax.vlines(line_data.index.values,
                      line_data['min'], line_data['max'],
                      lw=2, colors=self._forecolor)

        ax.set_yticks(np.arange(n_categories))
        ax.set_yticklabels(data.index.names)
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_frame_on(False)

    def plot_intersections(self, ax):
        """Plot bars indicating intersection size
        """
        ax.bar(np.arange(len(self.intersections)), self.intersections,
               width=.5, color=self._forecolor, zorder=10, align='center')
        ax.xaxis.set_visible(False)
        for x in ['top', 'bottom', 'right']:
            ax.spines[x].set_visible(False)
        ax.yaxis.grid(True)
        ax.set_ylabel('Intersection size')
        # ax.get_yaxis().set_tick_params(direction='in')

    def plot_totals(self, ax):
        """Plot bars indicating total set size
        """
        ax.barh(np.arange(len(self.totals.index.values)), self.totals,
                height=.5, color=self._forecolor, align='center')
        max_total = self.totals.max()
        ax.set_xlim(max_total, 0)
        for x in ['top', 'left', 'right']:
            ax.spines[x].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.grid(True)
        ax.ticklabel_format(axis='x')

    def plot(self, fig=None):
        """Draw all parts of the plot onto fig or a new figure

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Defaults to a new figure.

        Returns
        -------
        subplots : dict of matplotlib.axes.Axes
            Keys are 'matrix', 'intersections', 'totals'
        """
        if fig is None:
            fig = plt.figure(figsize=(10, 6))
        specs = self.make_grid(fig)
        matrix_ax = fig.add_subplot(specs['matrix'])
        self.plot_matrix(matrix_ax)
        inters_ax = fig.add_subplot(specs['intersections'], sharex=matrix_ax)
        self.plot_intersections(inters_ax, )
        totals_ax = fig.add_subplot(specs['totals'], sharey=matrix_ax)
        self.plot_totals(totals_ax)
        return {'matrix': matrix_ax,
                'intersections': inters_ax,
                'totals': totals_ax}


def plot(data, fig=None, **kwargs):
    """Make an UpSet plot of data on fig

    Parameters
    ----------
    data : pandas.Series
        Values for each set to plot.
        Should have multi-index where each level is binary,
        corresponding to set membership.
    fig : matplotlib.figure.Figure, optional
        Defaults to a new figure.
    kwargs
        Other arguments for :class:`UpSet`

    Returns
    -------
    subplots : dict of matplotlib.axes.Axes
        Keys are 'matrix', 'intersections', 'totals'
    """
    return UpSet(data, **kwargs).plot(fig)
