import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from upsetplot import generate_counts, generate_samples, query

# `query` is mostly tested through plotting tests, especially tests of
# `_process_data` which cover sort_by, sort_categories_by, subset_size
# and sum_over.


@pytest.mark.parametrize(
    "data",
    [
        generate_counts(),
        generate_samples(),
    ],
)
@pytest.mark.parametrize(
    "param_set",
    [
        [{"present": "cat1"}, {"absent": "cat1"}],
        [{"max_degree": 0}, {"min_degree": 1, "max_degree": 2}, {"min_degree": 3}],
        [{"max_subset_size": 30}, {"min_subset_size": 31}],
        [
            {"present": "cat1", "max_subset_size": 30},
            {"absent": "cat1", "max_subset_size": 30},
            {"present": "cat1", "min_subset_size": 31},
            {"absent": "cat1", "min_subset_size": 31},
        ],
    ],
)
def test_mece_queries(data, param_set):
    unfiltered_results = query(data)
    all_results = [query(data, **params) for params in param_set]

    # category_totals is unaffected by filter
    for results in all_results:
        assert_series_equal(unfiltered_results.category_totals, results.category_totals)

    combined_data = pd.concat([results.data for results in all_results])
    combined_data.sort_index(inplace=True)
    assert_frame_equal(unfiltered_results.data.sort_index(), combined_data)

    combined_sizes = pd.concat([results.subset_sizes for results in all_results])
    combined_sizes.sort_index(inplace=True)
    assert_series_equal(unfiltered_results.subset_sizes.sort_index(), combined_sizes)
