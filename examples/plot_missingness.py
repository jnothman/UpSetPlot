"""
=======================================
Plot the distribution of missing values
=======================================

UpSet plots are often used to show which variables are missing together.

Passing a callable ``indicators=pd.isna`` to :class:`UpSet` or :func:`plot` is
an easy way to categorise a record by the variables that are missing in it.
"""

from matplotlib import pyplot as plt
import pandas as pd
from upsetplot import plot

TITANIC_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'  # noqa
data = pd.read_csv(TITANIC_URL)

plot(data, indicators=pd.isna, show_counts=True)
plt.show()
