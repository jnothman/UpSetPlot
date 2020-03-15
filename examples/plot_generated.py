"""
============================
Plotting with generated data
============================

This example illustrates basic plotting functionality using generated data.
"""

from matplotlib import pyplot as plt
from upsetplot import generate_counts, plot

example = generate_counts()
print(example)

plot(example)
plt.suptitle('Ordered by degree')
plt.show()

plot(example, sort_by='cardinality')
plt.suptitle('Ordered by cardinality')
plt.show()

plot(example, show_counts='%d')
plt.suptitle('With counts shown')
plt.show()

plot(example, show_counts='%d', show_percentages=True)
plt.suptitle('With counts and % shown')
plt.show()
