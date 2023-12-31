__version__ = "0.10dev1"

import os

if os.environ.get("__IN-SETUP", None) != "1":
    from .data import (
        from_contents,
        from_indicators,
        from_memberships,
        generate_counts,
        generate_data,
        generate_samples,
    )
    from .plotting import UpSet, plot
    from .reformat import query

    __all__ = [
        "UpSet",
        "generate_data",
        "generate_counts",
        "generate_samples",
        "plot",
        "from_memberships",
        "from_contents",
        "from_indicators",
        "query",
    ]
