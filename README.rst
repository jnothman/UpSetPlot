UpSetPlot documentation
============================

|version| |licence| |py-versions|

|issues| |build| |docs| |coverage|

This is another Python implementation of UpSet plots by Lex et al. [Lex2014]_.

It tries to provide a simple, interface backed by an extensible,
object-oriented design.

The basic input format is a `pandas.Series` containing counts
corresponding to set intersection sizes.  The index indicates which rows
pertain to which sets, by having multiple boolean indices, like ``example``
in the following::

    >>> from upsetplot import generate_data
    >>> example = generate_data(aggregated=True)
    >>> example  # doctest: +NORMALIZE_WHITESPACE
    set0   set1   set2
    False  False  False      56
                  True      283
           True   False    1279
                  True     5882
    True   False  False      24
                  True       90
           True   False     429
                  True     1957
    Name: value, dtype: int64

Then::

    >>> from upsetplot import plot
    >>> plot(example)  # doctest: +SKIP

makes:

.. image:: http://upsetplot.readthedocs.io/en/latest/_images/sphx_glr_plot_generated_001.png

Why an alternative to py-upset?
-------------------------------

Probably for petty reasons. It appeared `py-upset
<https://github.com/ImSoErgodic/py-upset>`_ was not being maintained.  Its
input format was undocumented, inefficient and, IMO, inappropriate.  It did not
facilitate showing plots of each set intersection distribution as in Lex et
al's work introducing UpSet plots. Nor did it include the horizontal bar plots
illustrated there. It did not support Python 2. I decided it would be easier to
construct a cleaner version than to fix it.

References
----------

.. [Lex2014] Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot, Hanspeter Pfister,
   *UpSet: Visualization of Intersecting Sets*,
   IEEE Transactions on Visualization and Computer Graphics (InfoVis '14), vol. 20, no. 12, pp. 1983–1992, 2014.
   doi: `doi.org/10.1109/TVCG.2014.2346248 <https://doi.org/10.1109/TVCG.2014.2346248>`_


.. |py-versions| image:: https://img.shields.io/pypi/pyversions/upsetplot.svg
    :alt: Python versions supported

.. |version| image:: https://badge.fury.io/py/upsetplot.svg
    :alt: Latest version on PyPi
    :target: https://badge.fury.io/py/upsetplot

.. |build| image:: https://travis-ci.org/jnothman/UpSetPlot.svg?branch=master
    :alt: Travis CI build status
    :scale: 100%
    :target: https://travis-ci.org/jnothman/UpSetPlot

.. |issues| image:: https://img.shields.io/github/issues/jnothman/UpSetPlot.svg
    :alt: Issue tracker
    :target: https://github.com/jnothman/UpSetPlot

.. |coverage| image:: https://coveralls.io/repos/github/jnothman/UpSetPlot/badge.svg
    :alt: Test coverage
    :target: https://coveralls.io/github/jnothman/UpSetPlot

.. |docs| image:: https://readthedocs.org/projects/upsetplot/badge/?version=latest
     :alt: Documentation Status
     :scale: 100%
     :target: https://upsetplot.readthedocs.io/en/latest/?badge=latest

.. |licence| image:: https://img.shields.io/badge/Licence-BSD-blue.svg
     :target: https://opensource.org/licenses/BSD-3-Clause
