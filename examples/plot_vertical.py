"""
====================
Vertical orientation
====================

This illustrates the effect of orientation='vertical'.
"""

from matplotlib import pyplot as plt
from upsetplot import generate_counts, plot

example = generate_counts()
plot(example, orientation='vertical')
plt.suptitle('A vertical plot')
plt.show()

plot(example, orientation='vertical', show_counts='%d')
plt.suptitle('A vertical plot with counts shown')
plt.show()

plot(example, orientation='vertical', show_counts='%d', show_percentages=True)
plt.suptitle('With counts and percentages shown')
plt.show()
