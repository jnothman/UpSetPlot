"""
===================================
Basic: Examples with generated data
===================================

This example illustrates basic plotting functionality using generated data.
"""

import matplotlib
from matplotlib import pyplot as plt

from upsetplot import generate_counts, plot

example = generate_counts()
print(example)

##########################################################################

plot(example)
plt.suptitle("Ordered by degree")
plt.show()

##########################################################################

plot(example, sort_by="cardinality")
plt.suptitle("Ordered by cardinality")
plt.show()

##########################################################################

plot(example, show_counts="{:,}")
plt.suptitle("With counts shown, using a thousands separator")
plt.show()

##########################################################################

plot(example, show_counts="%d", show_percentages=True)
plt.suptitle("With counts and % shown")
plt.show()

##########################################################################

plot(example, show_percentages="{:.2%}")
plt.suptitle("With fraction shown in custom format")
plt.show()

##########################################################################

matplotlib.rcParams["font.size"] = 6
plot(example, show_percentages="{:.2%}")
plt.suptitle("With a smaller font size")
plt.show()
