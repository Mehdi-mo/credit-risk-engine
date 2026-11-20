"""How exactly a rerun has to match a saved file (slice 9).

On the machine that wrote the parquet files, a rerun reproduces them bit for
bit, and every check in this project asserts a difference of exactly 0.0.

On a DIFFERENT machine it does not, and slice 9 measured by how much: the
feature build differs in one column (log_annual_inc, 1.8e-15) and the
probabilities by 2.2e-16. Same code, same numpy and scikit-learn versions -
the maths library underneath (log1p, the matrix multiply) rounds the last bit
differently on Windows and on Linux. The Docker image is Linux, so this is
not a hypothetical.

So the tolerance is a setting, and the default is exact:

    CRE_FLOAT_TOL unset   -> 0.0     the machine that built the files
    CRE_FLOAT_TOL=1e-12   ->         any other machine, including the container

1e-12 is a thousand times the measured difference and a hundred million
times smaller than the third decimal of a probability. A real bug - a wrong
median, a reordered column - shows up at 1e-3 or larger, so it still fails.

ONE ROW AGAINST A BATCH
There is a second, smaller effect, and it does not need a second machine. The
same application scored on its own and scored as one row of a batch can differ
in the 16th digit (measured in slice 9 on Linux: at most 3.3e-16 over all
5,378 test rows, about a quarter of them not bit-identical; the closest any
test probability comes to the threshold is 2.8e-4, a trillion times further). The matrix multiply inside the model adds
71 numbers up in a different order for one row than for many. The service
scores one application at a time; the published file was scored as a batch.
So "one request reproduces the published probability" is asserted to
SINGLE_ROW_TOL, and "the batch reproduces it" to FLOAT_TOL.
"""

import os

FLOAT_TOL = float(os.environ.get('CRE_FLOAT_TOL', '0'))
SINGLE_ROW_TOL = max(FLOAT_TOL, 1e-12)


def max_abs_diff(a, b):
    """Largest absolute difference between two numeric arrays."""
    import numpy as np
    a, b = np.asarray(a, dtype='float64'), np.asarray(b, dtype='float64')
    assert a.shape == b.shape, f'shapes differ: {a.shape} vs {b.shape}'
    return float(np.abs(a - b).max()) if a.size else 0.0
