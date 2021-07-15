"""
====================
Changing Plot Colors
====================

This example illustrates use of matplotlib and upsetplot color settings, aside
from matplotlib style sheets.

Upsetplot provides some color settings:

* ``facecolor``: sets the color for intersection size bars, and for active
  matrix dots. Defaults to white on a dark background, otherwise black.
* ``other_dots_color``: sets the color for other (inactive) dots. Specify as a
  color, or a float specifying opacity relative to facecolor.
* ``shading_color``: sets the color odd rows. Specify as a color, or a float
  specifying opacity relative to facecolor.

For an introduction to matplotlib theming see:

* `Tutorial
  <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`__
* `Reference
  <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`__
"""

from matplotlib import pyplot as plt
from upsetplot import generate_counts, plot

example = generate_counts()

plot(example, facecolor="darkblue")
plt.suptitle('facecolor="darkblue"')
plt.show()

##########################################################################

plot(example, facecolor="darkblue", shading_color="lightgray")
plt.suptitle('facecolor="darkblue", shading_color="lightgray"')
plt.show()

##########################################################################

with plt.style.context('Solarize_Light2'):
    plot(example)
    plt.suptitle('matplotlib classic stylesheet')
    plt.show()

##########################################################################

with plt.style.context('dark_background'):
    plot(example, show_counts=True)
    plt.suptitle('matplotlib dark_background stylesheet')
    plt.show()

##########################################################################

with plt.style.context('dark_background'):
    plot(example, show_counts=True, shading_color=.15)
    plt.suptitle('matplotlib dark_background stylesheet, shading_color=.15')
    plt.show()

##########################################################################

with plt.style.context('dark_background'):
    plot(example, show_counts=True, facecolor="red")
    plt.suptitle('matplotlib dark_background, facecolor="red"')
    plt.show()

##########################################################################

with plt.style.context('dark_background'):
    plot(example, show_counts=True, facecolor="red", other_dots_color=.4,
         shading_color=.2)
    plt.suptitle('dark_background, red face, stronger other colors')
    plt.show()
