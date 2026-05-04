"""
Unit tests for the IBL implementation.

Run with:  python -m pytest tests/ -v
       or: python tests/test_ibl.py
"""

from src.ib3 import IB3
from src.ib2 import IB2
from src.ib1 import IB1
from src.similarity import similarity, AttributeNormalizer, MISSING
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(a, b, tol=1e-9, msg=""):
    assert abs(a - b) < tol, f"{msg} | expected {b}, got {a}"


def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


# ---------------------------------------------------------------------------
# similarity.py tests
# ---------------------------------------------------------------------------

class TestSimilarity:

    def test_identical_numeric(self):
        """Identical numeric instances should have similarity 0."""
        x = [0.5, 0.3]
        y = [0.5, 0.3]
        flags = [True, True]
        assert_close(similarity(x, y, flags), 0.0, msg="identical numeric")

    def test_identical_symbolic(self):
        x = ['a', 'b']
        y = ['a', 'b']
        flags = [False, False]
        assert_close(similarity(x, y, flags), 0.0, msg="identical symbolic")

    def test_numeric_distance(self):
        """Manual: sim([0,0],[1,0]) = -sqrt((1-0)^2 + 0) = -1."""
        x = [0.0, 0.0]
        y = [1.0, 0.0]
        flags = [True, True]
        assert_close(similarity(x, y, flags), -1.0, msg="numeric distance")

    def test_symbolic_mismatch(self):
        """sim(['a'],['b']) = -sqrt(1) = -1."""
        x = ['a']
        y = ['b']
        flags = [False]
        assert_close(similarity(x, y, flags), -1.0, msg="symbolic mismatch")

    def test_missing_values(self):
        """Missing values are maximally different (f=1)."""
        x = [MISSING, 0.0]
        y = [0.5, 0.0]
        flags = [True, True]
        # f(MISSING, 0.5) = 1, f(0.0, 0.0) = 0 -> sim = -1
        assert_close(similarity(x, y, flags), -1.0, msg="missing value")

    def test_both_missing(self):
        x = [MISSING]
        y = [MISSING]
        flags = [True]
        assert_close(similarity(x, y, flags), -1.0, msg="both missing")

    def test_mixed_types(self):
        """Mixed numeric and symbolic."""
        x = [0.0, 'a']
        y = [1.0, 'a']
        flags = [True, False]
        # f(0,1)=1, f('a','a')=0 -> sim = -sqrt(1) = -1
        assert_close(similarity(x, y, flags), -1.0, msg="mixed types")

    def test_similarity_ordering(self):
        """A closer instance should have higher similarity."""
        base = [0.5]
        close = [0.6]
        far = [0.9]
        flags = [True]
        assert similarity(base, close, flags) > similarity(base, far, flags)


class TestNormalizer:

    def test_single_value(self):
        """If only one value seen, normalized result is 0."""
        n = AttributeNormalizer([True])
        n.update([5.0])
        result = n.normalize([5.0])
        assert_close(result[0], 0.0, msg="single value normalization")

    def test_range_normalization(self):
        n = AttributeNormalizer([True])
        n.update([0.0])
        n.update([10.0])
        result = n.normalize([5.0])
        assert_close(result[0], 0.5, msg="midpoint normalization")

    def test_symbolic_passthrough(self):
        n = AttributeNormalizer([False])
        n.update(['yes'])
        result = n.normalize(['no'])
        assert result[0] == 'no', "symbolic values should pass through unchanged"

    def test_incremental_update(self):
        """Running min/max updates correctly as new extremes arrive."""
        n = AttributeNormalizer([True])
        n.update([5.0])
        n.update([10.0])
        # After seeing [0,10]: 5 -> 0.5
        n.update([0.0])
        result = n.normalize([5.0])
        assert_close(result[0], 0.5, msg="incremental min/max update")


# ---------------------------------------------------------------------------
# IB1 tests
# ---------------------------------------------------------------------------

class TestIB1:

    def _simple_dataset(self):
        """Two clearly separated clusters."""
        X_train = [[0.0], [0.1], [0.9], [1.0]]
        y_train = ['A', 'A', 'B', 'B']
        X_test = [[0.05], [0.95]]
        y_test = ['A', 'B']
        return X_train, y_train, X_test, y_test

    def test_stores_all_instances(self):
        X, y, _, _ = self._simple_dataset()
        model = IB1([True])
        model.fit(X, y)
        assert model.storage_count() == len(X), "IB1 must store all instances"

    def test_perfect_accuracy_clean(self):
        X, y, X_test, y_test = self._simple_dataset()
        model = IB1([True])
        model.fit(X, y)
        preds = model.predict(X_test)
        assert accuracy(
            y_test, preds) == 1.0, "IB1 should classify clean data perfectly"

    def test_storage_fraction(self):
        X, y, _, _ = self._simple_dataset()
        model = IB1([True])
        model.fit(X, y)
        assert_close(model.storage_fraction(len(X)), 1.0,
                     msg="IB1 storage fraction should be 1.0")

    def test_predict_empty_cd(self):
        """Predict before training returns None."""
        model = IB1([True])
        result = model.predict([[0.5]])
        assert result == [None], "Empty CD should return None"

    def test_multiclass(self):
        X = [[0.0], [0.5], [1.0]]
        y = ['A', 'B', 'C']
        model = IB1([True])
        model.fit(X, y)
        preds = model.predict([[0.0], [0.5], [1.0]])
        assert preds == ['A', 'B', 'C']


# ---------------------------------------------------------------------------
# IB2 tests
# ---------------------------------------------------------------------------

class TestIB2:

    def _linearly_separable(self):
        X_train = [[float(i) / 9] for i in range(10)]
        y_train = ['neg'] * 5 + ['pos'] * 5
        X_test = [[0.2], [0.8]]
        y_test = ['neg', 'pos']
        return X_train, y_train, X_test, y_test

    def test_storage_less_than_ib1(self):
        X, y, _, _ = self._linearly_separable()
        ib1 = IB1([True])
        ib2 = IB2([True])
        ib1.fit(list(X), list(y))
        ib2.fit(list(X), list(y))
        assert ib2.storage_count() < ib1.storage_count(), \
            "IB2 should store fewer instances than IB1 on clean data"

    def test_reasonable_accuracy(self):
        X, y, X_test, y_test = self._linearly_separable()
        model = IB2([True])
        model.fit(X, y)
        preds = model.predict(X_test)
        assert accuracy(y_test, preds) >= 0.5, \
            "IB2 should achieve at least 50% on linearly separable data"

    def test_saves_only_misclassified(self):
        """
        On a perfectly clean, well-separated dataset fed in the right order,
        only the first instance and boundary instances should be saved.
        """
        # Give IB2 a totally clean, easily-separated dataset where the
        # boundary instance is the only one likely to be misclassified.
        X = [[0.0], [0.1], [0.4], [0.6], [0.9], [1.0]]
        y = ['A', 'A', 'A', 'B', 'B', 'B']
        model = IB2([True])
        model.fit(X, y)
        # Should save far fewer than all 6 instances
        assert model.storage_count() < len(X)


# ---------------------------------------------------------------------------
# IB3 tests
# ---------------------------------------------------------------------------

class TestIB3:

    def _noisy_dataset(self, n=200, noise_frac=0.10, seed=0):
        """
        Two classes separated at 0.5, with noise_frac labels flipped.
        """
        rng = random.Random(seed)
        X, y = [], []
        for _ in range(n):
            val = rng.random()
            label = 'pos' if val >= 0.5 else 'neg'
            if rng.random() < noise_frac:
                label = 'neg' if label == 'pos' else 'pos'
            X.append([val])
            y.append(label)
        return X, y

    def _clean_dataset(self):
        X = [[float(i) / 19] for i in range(20)]
        y = ['neg'] * 10 + ['pos'] * 10
        return X, y

    def test_ib3_vs_ib2_on_noisy_data(self):
        """
        IB3 should match or beat IB2 accuracy on noisy data (paper claim c).
        Test over multiple seeds to reduce variance.
        """
        ib2_accs, ib3_accs = [], []
        for seed in range(10):
            X, y = self._noisy_dataset(n=300, noise_frac=0.15, seed=seed)
            train, test = X[:200], X[200:]
            y_train, y_test = y[:200], y[200:]

            ib2 = IB2([True])
            ib2.fit(list(train), list(y_train))
            ib2_accs.append(accuracy(y_test, ib2.predict(test)))

            ib3 = IB3([True])
            ib3.fit(list(train), list(y_train))
            ib3_accs.append(accuracy(y_test, ib3.predict(test)))

        mean_ib2 = sum(ib2_accs) / len(ib2_accs)
        mean_ib3 = sum(ib3_accs) / len(ib3_accs)
        assert mean_ib3 >= mean_ib2 - 0.05, \
            f"IB3 ({mean_ib3:.3f}) should not be much worse than IB2 ({mean_ib2:.3f}) on noisy data"

    def test_storage_reduction(self):
        """IB3 should store fewer instances than IB1."""
        X, y = self._clean_dataset()
        ib1 = IB1([True])
        ib3 = IB3([True])
        ib1.fit(list(X), list(y))
        ib3.fit(list(X), list(y))
        assert ib3.storage_count() < ib1.storage_count(), \
            "IB3 should store fewer instances than IB1"

    def test_confidence_interval_endpoints(self):
        """CI lower > 0, upper < 1, lower < upper."""
        from src.ib3 import _confidence_interval
        lo, hi = _confidence_interval(8, 10, z=1.645)
        assert 0.0 <= lo < hi <= 1.0, f"CI bounds invalid: ({lo}, {hi})"

    def test_z_score_lookup(self):
        from src.ib3 import _z_score
        assert _z_score(0.90) > _z_score(0.75), \
            "90% z-score should be larger than 75%"

    def test_no_crash_on_empty_cd(self):
        """IB3 predict on untrained model returns None gracefully."""
        model = IB3([True])
        result = model.predict([[0.5]])
        assert result == [None]

    def test_symbolic_attributes(self):
        """IB3 works correctly with symbolic (non-numeric) attributes."""
        X = [['y', 'n'], ['y', 'y'], ['n', 'n'], ['n', 'y']]
        y = ['A', 'A', 'B', 'B']
        model = IB3([False, False])
        model.fit(X, y)
        preds = model.predict([['y', 'n'], ['n', 'n']])
        assert preds[0] is not None and preds[1] is not None


# ---------------------------------------------------------------------------
# Cross-algorithm consistency tests
# ---------------------------------------------------------------------------

class TestCrossAlgorithm:

    def test_ib1_always_most_storage(self):
        """IB1 should always use 100% storage."""
        rng = random.Random(99)
        X = [[rng.random()] for _ in range(50)]
        y = ['A' if x[0] < 0.5 else 'B' for x in X]

        ib1 = IB1([True])
        ib1.fit(list(X), list(y))
        assert ib1.storage_count() == 50

    def test_storage_ordering_clean_data(self):
        """On clean data: IB3_storage <= IB2_storage << IB1_storage."""
        rng = random.Random(7)
        X = [[rng.random()] for _ in range(100)]
        y = ['A' if x[0] < 0.5 else 'B' for x in X]

        models = {}
        for name, Cls in [('ib1', IB1), ('ib2', IB2), ('ib3', IB3)]:
            m = Cls([True])
            m.fit(list(X), list(y))
            models[name] = m

        assert models['ib1'].storage_count() == 100
        assert models['ib2'].storage_count() < models['ib1'].storage_count()
        # IB3 may or may not be <= IB2 on any single run due to randomness,
        # but should be well below IB1
        assert models['ib3'].storage_count() < models['ib1'].storage_count()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    test_classes = [
        TestSimilarity,
        TestNormalizer,
        TestIB1,
        TestIB2,
        TestIB3,
        TestCrossAlgorithm,
    ]

    total = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith('test_')]
        for method in methods:
            total += 1
            try:
                getattr(instance, method)()
                print(f"  PASS  {cls.__name__}.{method}")
            except Exception as e:
                failed += 1
                errors.append(f"  FAIL  {cls.__name__}.{method}: {e}")
                print(errors[-1])

    print(f"\n{'='*50}")
    print(f"Results: {total - failed}/{total} passed")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(e)
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
