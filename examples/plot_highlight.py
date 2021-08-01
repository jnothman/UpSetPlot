"""
=============================
Highlighting selected subsets
=============================

"""

from upsetplot import generate_counts, UpSet

example = generate_counts()

upset = UpSet(example)
upset.style_subsets("cat2", facecolor="blue")
upset.plot()

##########################################################################

upset = UpSet(example, orientation="vertical")
upset.style_subsets("cat2", "cat1", edgecolor="red", linewidth=2)
upset.plot()

##########################################################################

upset = UpSet(example)
upset.style_subsets("cat2", facecolor="lightblue", hatch="/")
upset.plot()
