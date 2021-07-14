"""
====================
Changing Plot Colors
====================

This example illustrates use of matplotlib and upsetplot color settings.
"""

from matplotlib import pyplot as plt
from upsetplot import generate_counts, plot

example = generate_counts()

plot(example, facecolor="red")
plt.suptitle('Setting facecolor')
plt.show()

plot(example, facecolor="red", shading_base_color="black")
plt.suptitle('Setting facecolor and shading_base_color')
plt.show()


with plt.style.context('classic'):
    plot(example)
    plt.suptitle('Using matplotlib classic stylesheet')
    plt.show()

with plt.style.context('dark_background'):
    plot(example, show_counts=True)
    plt.suptitle('Using matplotlib dark_background stylesheet')
    plt.show()
