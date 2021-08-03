"""
=============================
Highlighting selected subsets
=============================

Demonstrates use of the `style_subsets` method to mark some subsets as
different.

"""

from matplotlib import pyplot as plt
from upsetplot import generate_counts, UpSet

example = generate_counts()

upset = UpSet(example)
upset.style_subsets(include=["cat1", "cat2"],
                    facecolor="blue",
                    label="special")
upset.plot()
plt.suptitle("Paint blue subsets including both cat1 and cat2; show a legend")
plt.show()

##########################################################################

upset = UpSet(example, orientation="vertical")
upset.style_subsets(include="cat2", exclude="cat1", edgecolor="red",
                    linewidth=2)
upset.plot()
plt.suptitle("Border for subsets including cat2 but not cat1")
plt.show()

##########################################################################

upset = UpSet(example)
upset.style_subsets(min_subset_size=1000,
                    facecolor="lightblue", hatch="xx",
                    label="big")
upset.plot()
plt.suptitle("Hatch subsets with size >1000")
plt.show()
