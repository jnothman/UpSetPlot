from __future__ import print_function, division, absolute_import

import pandas as pd
import numpy as np


def generate_data(seed=0, n_samples=10000, n_sets=3, aggregated=False):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({'value': np.zeros(n_samples)})
    for i in range(n_sets):
        r = rng.rand(n_samples)
        df['set%d' % i] = r > rng.rand()
        df['value'] += r

    df.set_index(['set%d' % i for i in range(n_sets)], inplace=True)
    if aggregated:
        return df.value.groupby(level=list(range(n_sets))).count()
    return df.value
