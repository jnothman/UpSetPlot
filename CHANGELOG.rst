What's new in version 0.2
-------------------------

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
