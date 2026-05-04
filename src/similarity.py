"""
Similarity function as defined in Aha, Kibler & Albert (1991).

Similarity(x, y) = -sqrt( sum_i f(x_i, y_i) )

where:
  f(x_i, y_i) = (x_i - y_i)^2   for numeric attributes  (after normalization)
  f(x_i, y_i) = 0 if x_i == y_i, else 1   for boolean/symbolic attributes
  f(x_i, y_i) = 1 if either value is missing (maximally different)
  f(x_i, y_i) = 1 if both values are missing

Attributes are normalized by tracking the running min/max seen so far.
The similarity is negative so that "more similar" => "higher value",
consistent with the paper's argmax formulation.
"""


MISSING = None  # sentinel for missing attribute values


def feature_diff(xi, yi, is_numeric):
    """
    Compute f(xi, yi) for a single attribute.
    Assumes numeric attributes have already been normalized to [0, 1].
    """
    if xi is MISSING or yi is MISSING:
        return 1.0
    if is_numeric:
        return (xi - yi) ** 2
    else:
        return 0.0 if xi == yi else 1.0


def similarity(x, y, is_numeric_flags):
    """
    Compute similarity between two instances x and y.

    Parameters
    ----------
    x, y : list
        Attribute values (excluding class label). Numeric values must already
        be normalized to [0, 1] before calling this function.
    is_numeric_flags : list of bool
        True for each attribute that is numeric-valued.

    Returns
    -------
    float
        Similarity score in (-inf, 0]. Higher is more similar.
    """
    assert len(x) == len(y) == len(is_numeric_flags), (
        f"Length mismatch: x={len(x)}, y={len(y)}, flags={len(is_numeric_flags)}"
    )
    total = sum(
        feature_diff(xi, yi, num)
        for xi, yi, num in zip(x, y, is_numeric_flags)
    )
    return -(total ** 0.5)


class AttributeNormalizer:
    """
    Tracks running min/max for each numeric attribute and normalizes values
    incrementally, as the paper requires (normalization happens on the fly
    during incremental training).
    """

    def __init__(self, is_numeric_flags):
        self.is_numeric = is_numeric_flags
        n = len(is_numeric_flags)
        self.min_vals = [float('inf')] * n
        self.max_vals = [float('-inf')] * n

    def update(self, instance):
        """Update running min/max with a new instance's attribute values."""
        for i, (val, is_num) in enumerate(zip(instance, self.is_numeric)):
            if is_num and val is not MISSING:
                if val < self.min_vals[i]:
                    self.min_vals[i] = val
                if val > self.max_vals[i]:
                    self.max_vals[i] = val

    def normalize(self, instance):
        """
        Return a normalized copy of the instance.
        Numeric attributes scaled to [0, 1] using running min/max.
        If min == max (only one value seen), normalized value is 0.
        """
        normalized = []
        for i, (val, is_num) in enumerate(zip(instance, self.is_numeric)):
            if not is_num or val is MISSING:
                normalized.append(val)
            else:
                lo, hi = self.min_vals[i], self.max_vals[i]
                if hi == lo:
                    normalized.append(0.0)
                else:
                    normalized.append((val - lo) / (hi - lo))
        return normalized
