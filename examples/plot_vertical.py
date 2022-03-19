"""
====================
Vertical orientation
====================

This illustrates the effect of orientation='vertical'.
"""

from matplotlib import pyplot as plt
from upsetplot import generate_counts, plot, plotting

example = generate_counts()
plot(example, orientation='vertical')
plt.suptitle('A vertical plot')
plt.show()

##########################################################################

plot(example, orientation='vertical', show_counts='%d')
plt.suptitle('A vertical plot with counts shown')
plt.show()

##########################################################################

plot(example, orientation='vertical', show_counts='%d', show_percentages=True)
plt.suptitle('With counts and percentages shown')
plt.show()

#########################################################################
"""
    An UpSetplot with additional plots on vertical
    and tuning some visual parameters
"""
example = generate_counts(extra_columns=2)
fig = plotting.UpSet(example, orientation='vertical',
                     show_counts=True, facecolor="grey",
                     element_size=75)
fig.add_catplot('swarm', 'value', palette='colorblind')
fig.add_catplot('swarm', 'value1', palette='colorblind')
fig.add_catplot('swarm', 'value2', palette='colorblind')
fig.plot()
plt.show()
