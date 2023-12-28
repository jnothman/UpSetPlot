"""
======================================
Hiding subsets based on size or degree
======================================

This illustrates the use of ``min_subset_size``, ``max_subset_size``,
``min_degree`` or ``max_degree``.
"""

from matplotlib import pyplot as plt
from upsetplot import generate_counts, plot

example = generate_counts()

plot(example, show_counts=True)
plt.suptitle("Nothing hidden")
plt.show()

##########################################################################

plot(example, show_counts=True, min_subset_size=100)
plt.suptitle("Small subsets hidden")
plt.show()

##########################################################################

plot(example, show_counts=True, max_subset_size=500)
plt.suptitle("Large subsets hidden")
plt.show()

##########################################################################

plot(example, show_counts=True, min_degree=2)
plt.suptitle("Degree <2 hidden")
plt.show()

##########################################################################

plot(example, show_counts=True, max_degree=2)
plt.suptitle("Degree >2 hidden")
plt.show()
