"""
IB3 Algorithm — Aha, Kibler & Albert (1991), Table 5.

Extension of IB2 that adds a "wait and see" selective utilization filter.
Each saved instance maintains a classification record (correct/incorrect
attempts). A confidence-intervals-of-proportions test determines:

  - ACCEPTABLE: instance accuracy interval lower bound > class frequency
                interval upper bound  (z = 90%, paper default)
  - NOISY/DROP: instance accuracy interval upper bound < class frequency
                interval lower bound  (z = 75%, paper default)
  - MEDIOCRE:   intervals overlap -> wait for more evidence

Only ACCEPTABLE instances are used for classification decisions.
NOISY instances are removed from the concept description.

Classification record updates: for each training instance t, records are
updated for all saved instances within the hyper-sphere defined by the
distance to t's nearest acceptable neighbor. If no acceptable instances
exist yet, a random radius is chosen (see paper Section 4.1).
"""

import math
import random
from .similarity import similarity, AttributeNormalizer


# ---------------------------------------------------------------------------
# Confidence interval helpers
# ---------------------------------------------------------------------------

def _z_score(confidence):
    """
    Return the z-score for a two-sided confidence interval.
    Uses a lookup table for the values the paper specifies.
    """
    # Paper uses z=90% for acceptance, z=75% for dropping.
    # These correspond to one-sided z values (upper tail).
    table = {
        0.75: 1.150,   # 75% two-sided CI  -> z ≈ 1.150
        0.90: 1.645,   # 90% two-sided CI  -> z ≈ 1.645
        0.95: 1.960,
        0.99: 2.576,
    }
    if confidence in table:
        return table[confidence]
    # Fallback: rough approximation via normal quantile
    # Not needed for the paper's parameters but keeps the code general.
    raise ValueError(f"Unsupported confidence level: {confidence}. "
                     f"Choose from {list(table.keys())}")


def _confidence_interval(successes, trials, z):
    """
    Wilson score confidence interval for a proportion.
    Returns (lower, upper).

    Using Wilson interval (more robust than normal approximation for small n).
      centre = (successes + z²/2) / (trials + z²)
      half   = z * sqrt(p_hat*(1-p_hat)/n + z²/(4n²)) / (1 + z²/n)
    """
    if trials == 0:
        return (0.0, 1.0)
    n = trials
    z2 = z * z
    p_hat = successes / n
    centre = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    half = (z / (1 + z2 / n)) * math.sqrt(
        p_hat * (1 - p_hat) / n + z2 / (4 * n * n)
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Per-instance record
# ---------------------------------------------------------------------------

class _InstanceRecord:
    """
    Stores a saved instance and its classification performance record.
    Raw attribute values are stored; normalization happens at query time.
    """
    __slots__ = ('raw_attrs', 'label', 'correct', 'attempts')

    def __init__(self, raw_attrs, label):
        self.raw_attrs = raw_attrs
        self.label = label
        self.correct = 0
        self.attempts = 0

    def update(self, was_correct):
        self.attempts += 1
        if was_correct:
            self.correct += 1


# ---------------------------------------------------------------------------
# IB3
# ---------------------------------------------------------------------------

class IB3:
    """
    IB3: noise-tolerant, storage-reducing IBL algorithm.

    Parameters
    ----------
    is_numeric_flags : list of bool
        One entry per predictor attribute; True means numeric.
    z_accept : float
        Confidence level for the acceptance test (paper default: 0.90).
    z_drop : float
        Confidence level for the dropping test (paper default: 0.75).
    """

    def __init__(self, is_numeric_flags, z_accept=0.90, z_drop=0.75):
        self.is_numeric = is_numeric_flags
        self.normalizer = AttributeNormalizer(is_numeric_flags)
        self.z_accept = _z_score(z_accept)
        self.z_drop = _z_score(z_drop)

        # concept description: list of _InstanceRecord
        self.cd = []

        # class frequency tracking: {label -> count}, total count
        self._class_counts = {}
        self._total_seen = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Incrementally train IB3.

        Raw attribute values are stored in each _InstanceRecord.
        Normalization is applied at query time so that expanding ranges
        are always reflected correctly in distance computations.

        Returns
        -------
        train_accuracy : float
        """
        correct_count = 0

        for attrs, label in zip(X, y):
            # 1. Update normalizer and class frequency tracker
            self.normalizer.update(attrs)
            norm_query = self.normalizer.normalize(attrs)
            self._class_counts[label] = self._class_counts.get(label, 0) + 1
            self._total_seen += 1

            # Helper: normalized similarity from this query to a record
            def sim_to(rec):
                return similarity(
                    norm_query,
                    self.normalizer.normalize(rec.raw_attrs),
                    self.is_numeric
                )

            # 2. Find all acceptable instances and the nearest one
            acceptable = [r for r in self.cd if self._is_acceptable(r)]

            if acceptable:
                nearest_acc = max(acceptable, key=sim_to)
                nearest_acc_sim = sim_to(nearest_acc)
                pred = nearest_acc.label
                update_radius = -nearest_acc_sim  # positive distance
            else:
                if self.cd:
                    ranked = sorted(self.cd, key=sim_to, reverse=True)
                    r_idx = random.randint(0, len(ranked) - 1)
                    pred = ranked[r_idx].label
                    dist_nearest = -sim_to(ranked[0])
                    dist_farthest = -sim_to(ranked[-1])
                    if dist_nearest == dist_farthest:
                        update_radius = dist_nearest
                    else:
                        update_radius = random.uniform(
                            dist_nearest, dist_farthest)
                else:
                    pred = None
                    update_radius = None

            if pred == label:
                correct_count += 1

            # 3. Save if misclassified — store raw values
            if pred != label:
                self.cd.append(_InstanceRecord(attrs, label))

            # 4. Update classification records for instances within radius,
            #    then drop noisy ones
            if update_radius is not None:
                to_drop = []
                for rec in self.cd:
                    dist = -sim_to(rec)
                    if dist <= update_radius:
                        was_correct = (rec.label == label)
                        rec.update(was_correct)
                        if self._is_noisy(rec):
                            to_drop.append(rec)
                if to_drop:
                    drop_set = set(id(r) for r in to_drop)
                    self.cd = [r for r in self.cd if id(r) not in drop_set]

        n = len(X)
        return correct_count / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Predict using only ACCEPTABLE instances.
        If none are acceptable, falls back to the full CD (nearest neighbor).
        """
        predictions = []
        for attrs in X:
            pred = self._classify(attrs)
            predictions.append(pred)
        return predictions

    def _classify(self, raw_attrs):
        norm_query = self.normalizer.normalize(raw_attrs)
        acceptable = [r for r in self.cd if self._is_acceptable_any_class(r)]
        pool = acceptable if acceptable else self.cd
        if not pool:
            return None
        best_sim = float('-inf')
        best_label = None
        for rec in pool:
            norm_stored = self.normalizer.normalize(rec.raw_attrs)
            sim = similarity(norm_query, norm_stored, self.is_numeric)
            if sim > best_sim:
                best_sim = sim
                best_label = rec.label
        return best_label

    # ------------------------------------------------------------------
    # Significance tests (paper Section 4.1)
    # ------------------------------------------------------------------

    def _class_frequency_interval(self, label, z):
        """
        Confidence interval for the observed relative frequency of `label`.
        Computed over all training instances seen so far.
        """
        count = self._class_counts.get(label, 0)
        return _confidence_interval(count, self._total_seen, z)

    def _accuracy_interval(self, record, z):
        """Confidence interval for the instance's classification accuracy."""
        return _confidence_interval(record.correct, record.attempts, z)

    def _is_acceptable(self, record):
        """
        True if instance accuracy CI lower bound > class frequency CI upper bound.
        Uses z_accept for both intervals (conservative acceptance).
        """
        if record.attempts == 0:
            return False
        acc_lo, _ = self._accuracy_interval(record, self.z_accept)
        _, freq_hi = self._class_frequency_interval(
            record.label, self.z_accept)
        return acc_lo > freq_hi

    def _is_acceptable_any_class(self, record):
        """
        Acceptability check for prediction time (no current label context).
        Uses the instance's own stored label for frequency comparison.
        """
        if record.attempts == 0:
            return False
        acc_lo, _ = self._accuracy_interval(record, self.z_accept)
        _, freq_hi = self._class_frequency_interval(
            record.label, self.z_accept)
        return acc_lo > freq_hi

    def _is_noisy(self, record):
        """
        True if instance accuracy CI upper bound < class frequency CI lower bound.
        Uses z_drop (looser bound — drops more aggressively).
        """
        if record.attempts == 0:
            return False
        _, acc_hi = self._accuracy_interval(record, self.z_drop)
        freq_lo, _ = self._class_frequency_interval(record.label, self.z_drop)
        return acc_hi < freq_lo

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def storage_count(self):
        return len(self.cd)

    def storage_fraction(self, n_train):
        return self.storage_count() / n_train if n_train > 0 else 0.0

    def acceptable_count(self):
        return sum(1 for r in self.cd if self._is_acceptable_any_class(r))
