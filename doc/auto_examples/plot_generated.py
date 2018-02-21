"""Plotting with generated data
"""

from matplotlib import pyplot as plt
from upsetplot import generate_data, plot

example = generate_data(aggregated=True)
print(example)

plot(example)
plt.show()

plot(example, order='size')
plt.show()
