"""
============================
Plotting with generated data
============================

This example illustrates basic plotting functionality using generated data.
"""

from matplotlib import pyplot as plt
from upsetplot import generate_data, plot

example = generate_data(aggregated=True)
print(example)

plot(example)
plt.title('Ordered by degree')
plt.show()

plot(example, sort_by='cardinality')
plt.title('Ordered by cardinality')
plt.show()

plot(example, show_counts='%d')
plt.title('With counts shown')
plt.show()
