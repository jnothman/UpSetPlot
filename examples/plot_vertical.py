"""
====================
Vertical orientation
====================

This illustrates the effect of orientation='vertical'.
"""

from matplotlib import pyplot as plt
from upsetplot import generate_data, plot

example = generate_data(aggregated=True)
plot(example, orientation='vertical')
plt.title('A vertical plot')
plt.show()

plot(example, orientation='vertical', size_fmt='%d')
plt.title('A vertical plot with sizes')
plt.show()
