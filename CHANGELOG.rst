What's new in version 0.6.1
---------------------------

Fix for latest versions of setuptools, thanks to :user:`Marius Bakke <mbakke>`

What's new in version 0.6
-------------------------

- Added `add_stacked_bars`, similar to `add_catplot` but to add stacked bar
  charts to show discrete variable distributions within each subset.
  (:issue:`137`)
- Improved ability to control colors, and added a new example of same.
  Parameters ``other_dots_color`` and ``shading_color`` were added.
  ``facecolor`` will now default to white if
  ``matplotlib.rcParams['axes.facecolor']`` is dark. (:issue:`138`)
- Added `style_subsets` to colour intersection size bars and matrix
  dots in the plot according to a specified query. (:issue:`152`)
- Added `from_indicators` to allow yet another data input format. This
  allows category membership to be easily derived from a DataFrame, such as
  when plotting missing values in the columns of a DataFrame. (:issue:`143`)

What's new in version 0.5
-------------------------

- Support using input intersection order with ``sort_by=None`` (:issue:`133`
  with thanks to :user:`Brandon B <outlace>`).
- Add parameters for filtering by subset size (with thanks to
  :user:`Sichong Peng <SichongP>`) and degree. (:issue:`134`)
- Fixed an issue where tick labels were not given enough space and overlapped
  category totals. (:issue:`132`)
- Fixed an issue where our implementation of ``sort_by='degree'`` apparently
  gave incorrect results for some inputs and versions of Pandas. (:issue:`134`)

What's new in version 0.4.4
---------------------------

- Fixed a regresion which caused the first column to be hidden
  (:issue:`125`)

What's new in version 0.4.3
---------------------------

- Fixed issue with the order of catplots being reversed for vertical plots
  (:issue:`122` with thanks to :user:`Enrique Fernandez-Blanco <ennanco>`)
- Fixed issue with the x limits of vertical plots (:issue:`121`).

What's new in version 0.4.2
---------------------------

- Fixed large x-axis plot margins with high number of unique intersections
  (:issue:`106` with thanks to :user:`Yidi Huang <huangy6>`)

What's new in version 0.4.1
---------------------------

- Fixed the calculation of percentage which was broken in 0.4.0. (:issue:`101`)

What's new in version 0.4
-------------------------

- Added option to display both the absolute frequency and the percentage of
  the total for each intersection and category. (:issue:`89` with thanks to
  :user:`Carlos Melus <maziello>` and :user:`Aaron Rosenfeld <arosenfeld>`)
- Improved efficiency where there are many categories, but valid combinations
  are sparse, if `sort_by='degree'`. (:issue:`82`)
- Permit truthy (not necessarily bool) values in index.
  (:issue:`74` with thanks to :user:`ZaxR`)
- `intersection_plot_elements` can now be set to 0 to hide the intersection
  size plot when `add_catplot` is used. (:issue:`80`)

What's new in version 0.3
-------------------------

- Added `from_contents` to provide an alternative, intuitive way of specifying
  category membership of elements.
- To improve code legibility and intuitiveness, `sum_over=False` was deprecated
  and a `subset_size` parameter was added.  It will have better default
  handling of DataFrames after a short deprecation period.
- `generate_data` has been replaced with `generate_counts` and
  `generate_samples`.
- Fixed the display of the "intersection size" label on plots, which had been
  missing.
- Trying to improve nomenclature, upsetplot now avoids "set" to refer to the
  top-level sets, which are now to be known as "categories". This matches the
  intuition that categories are named, logical groupings, as opposed to
  "subsets". To this end:

  - `generate_counts` (formerly `generate_data`) now names its categories
    "cat1", "cat2" etc. rather than "set1", "set2", etc.
  - the `sort_sets_by` parameter has been renamed to `sort_categories_by` and
    will be removed in version 0.4.

What's new in version 0.2.1
---------------------------

- Return a Series (not a DataFrame) from `from_memberships` if data is
  1-dimensional.

What's new in version 0.2
-------------------------

- Added `from_memberships` to allow a more convenient data input format.
- `plot` and `UpSet` now accept a `pandas.DataFrame` as input, if the
  `sum_over` parameter is also given.
- Added an `add_catplot` method to `UpSet` which adds Seaborn plots of set
  intersection data to show more than just set size or total.
- Shading of subset matrix is continued through to totals.
- Added a `show_counts` option to show counts at the ends of bar plots.
  (:issue:`5`)
- Defined `_repr_html_` so that an `UpSet` object will render in Jupyter
  notebooks.
  (:issue:`36`)
- Fix a bug where an error was raised if an input set was empty.
