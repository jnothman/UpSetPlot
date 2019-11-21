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

# specify barplot colors and size
plot(example, sort_by='cardinality',
     intersection_facecolor="purple",
     intersection_width=1.0,
     totals_facecolor="brown",
     totals_width=0.2)
plt.suptitle('Barplot colors and width')
plt.show()

# color by row
plot(example, sort_by='cardinality',
     color_by_row={"cat1": "orange", "cat2": "green"})
plt.suptitle('Colored by row')
plt.show()

# color by row, and color the barplot
plot(example, sort_by='cardinality',
     color_by_row={"cat1": "orange", "cat2": "green"},
     totals_as_row=True)
plt.suptitle('Colored by row (and barplot)')
plt.show()

# color by col
plot(example, sort_by='cardinality',
     color_by_col={("cat0", "cat1"): "orange"})
plt.suptitle('Ordered by cardinality')
plt.suptitle('Colored by column')
plt.show()

# color by coland color the barplot
plot(example, sort_by='cardinality',
     color_by_col={("cat0", "cat1"): "orange"},
     intersection_as_col=True)
plt.suptitle('Ordered by cardinality')
plt.suptitle('Colored by column (and barplot)')
plt.show()

# modify matrix lines
plot(example, sort_by='cardinality',
     matrix_line_color="darkgray",
     matrix_line_width=10)
plt.suptitle('Ordered by cardinality')
plt.suptitle('Matrix lines')
plt.show()

# modify dots color
plot(example, sort_by='cardinality',
     empty_dot_color="lightgreen",
     dot_color="darkgreen")
plt.suptitle('Ordered by cardinality')
plt.suptitle('Empty dot colors')
plt.show()

# to test if orientation change works
plot(example, sort_by='cardinality',
     intersection_facecolor="purple",
     intersection_width=1.0,
     totals_facecolor="brown",
     totals_width=0.2,
     empty_dot_color="lightgreen",
     matrix_line_color="darkgray",
     matrix_line_width=10,
     dot_color="darkgreen",
     orientation="verticle")
plt.suptitle('verticle')
plt.show()
