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
plt.show()
