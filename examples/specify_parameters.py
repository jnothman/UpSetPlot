"""
============================
    Specify parameters
============================

by JH Liu
"""

from matplotlib import pyplot as plt
from upsetplot import generate_counts, plot

example = generate_counts()
print(example)

# provide kws matrix
plot(example, sort_by='cardinality',
     color_by_row={"cat1": "orange", "cat2": "green"},
     matrix_kws={"s": 1}, lines_kws={"colors": "green", "linestyles": "--"})
plt.suptitle('Ordered by cardinality')
plt.show()

# provide kws bars
plot(example, sort_by='cardinality',
     color_by_row={"cat1": "orange", "cat2": "green"},
     intersection_kws={"width": 1.0},
     totals_kws={"color": "purple", "width": 0.2})
plt.suptitle('Ordered by cardinality')
plt.show()

# color by row
plot(example, sort_by='cardinality',
     color_by_row={"cat1": "orange", "cat2": "green"})
plt.suptitle('Ordered by cardinality')
plt.show()

# color by row, and color the barplot
plot(example, sort_by='cardinality',
     color_by_row={"cat1": "orange", "cat2": "green"},
     totals_as_row=True)
plt.suptitle('Ordered by cardinality')
plt.show()

# color by col
plot(example, sort_by='cardinality', color_by_col={("cat0", "cat1"): "orange"})
plt.suptitle('Ordered by cardinality')
plt.show()

# color by coland color the barplot
plot(example, sort_by='cardinality', color_by_col={("cat0", "cat1"): "orange"},
     intersection_as_col=True)
plt.suptitle('Ordered by cardinality')
plt.show()

# modify lines
plot(example, sort_by='cardinality', lines_color="darkgray", lines_width=10)
plt.suptitle('Ordered by cardinality')
plt.show()

# grids
plot(example, sort_by='cardinality',
     intersection_grids=False, totals_grids=False)
plt.suptitle('Ordered by cardinality')
plt.show()
