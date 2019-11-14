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

# color by row
plot(example, sort_by='cardinality', color_by_row={"cat1":"orange", "cat2":"green"})
plt.suptitle('Ordered by cardinality')
plt.show()

# color by row, and color the barplot
plot(example, sort_by='cardinality', color_by_row={"cat1":"orange", "cat2":"green"}, total_as_row=True)
plt.suptitle('Ordered by cardinality')
plt.show()

# color by col
plot(example, sort_by='cardinality', color_by_col={("cat0","cat1"):"orange"})
plt.suptitle('Ordered by cardinality')
plt.show()

# color by coland color the barplot
plot(example, sort_by='cardinality', color_by_col={("cat0","cat1"):"orange"}, intersection_as_col=True)
plt.suptitle('Ordered by cardinality')
plt.show()

# modify lines
plot(example, sort_by='cardinality', lines_color="darkgray", lines_width=10)
plt.suptitle('Ordered by cardinality')
plt.show()

# boundary
plot(example, sort_by='cardinality', lines_color="darkgray", min = 100, max=1000)
plt.suptitle('Ordered by cardinality')
plt.show()

# grids
plot(example, sort_by='cardinality', intersection_grids=False, total_grids=False)
plt.suptitle('Ordered by cardinality')
plt.show()