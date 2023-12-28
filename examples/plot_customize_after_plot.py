"""
======================================
Customizing axis labels after plotting
======================================

This example illustrates how the return value of the plot method can be used
to customize aspects of the plot, such as axis labels.
"""

from upsetplot import generate_counts, plot

example = generate_counts()
print(example)

##########################################################################

plot_result = plot(example)
plot_result["intersection"].set_ylabel("Subset cardinality")
plot_result["matrix"].set_xlabel("Subsets")
plt.show()
