What's new in version 0.3
-------------------------

- Added `from_contents` to provide an alternative, intuitive way of specifying
  category membership of elements.
- Fixed the display of the "intersection size" label on plots, which had been
  missing.
- Trying to improve nomenclature, upsetplot now avoids "set" to refer to the
  top-level sets, which are now to be known as "categories". This matches the
  intuition that categories are named, logical groupings, as opposed to
  "subsets". To this end, the `sort_sets_by` parameter has been renamed to
  `sort_categories_by` and will be removed in version 0.5.

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
