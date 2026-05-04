"""
IB1 Algorithm — Aha, Kibler & Albert (1991), Table 1.

The simplest instance-based learning algorithm. Stores every training
instance and classifies new instances by nearest-neighbor lookup.

CD <- {}
for each x in TrainingSet:
    1. compute Similarity(x, y) for each y in CD
    2. ymax <- y in CD with maximal Similarity
    3. if class(x) == class(ymax): correct, else: incorrect
    4. CD <- CD ∪ {x}       # always save
"""

from .similarity import similarity, AttributeNormalizer


class IB1:
    """
    IB1: nearest-neighbor classifier that saves every training instance.

    Parameters
    ----------
    is_numeric_flags : list of bool
        One entry per predictor attribute; True means numeric.
    """

    def __init__(self, is_numeric_flags):
        self.is_numeric = is_numeric_flags
        self.normalizer = AttributeNormalizer(is_numeric_flags)
        # Each entry: (normalized_attributes, class_label)
        self.concept_description = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Incrementally train on a sequence of instances.

        Parameters
        ----------
        X : list of lists  — attribute vectors (raw, un-normalized)
        y : list           — class labels

        Returns
        -------
        train_accuracy : float
            Online accuracy during training (fraction of instances that were
            correctly classified before being added to CD).

        Note on normalization:
            Raw attribute values are stored in the concept description.
            Normalization happens at classification time using the running
            min/max seen so far. This ensures that when new extremes arrive
            and re-scale the range, all stored instances are compared on the
            same up-to-date scale.
        """
        correct = 0
        for attrs, label in zip(X, y):
            # Update running min/max with the new instance's raw values
            self.normalizer.update(attrs)

            if self.concept_description:
                pred = self._classify(attrs)
                if pred == label:
                    correct += 1

            # Store raw values — normalization applied at query time
            self.concept_description.append((attrs, label))

        n = len(X)
        train_accuracy = correct / n if n > 0 else 0.0
        return train_accuracy

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Predict class labels for a list of instances.

        Parameters
        ----------
        X : list of lists — raw attribute vectors

        Returns
        -------
        list of predicted class labels
        """
        predictions = []
        for attrs in X:
            pred = self._classify(attrs)
            predictions.append(pred)
        return predictions

    def _classify(self, raw_attrs):
        """
        Return the class of the most similar instance in CD.
        Both the query and all stored instances are normalized using the
        current running min/max before comparison.
        """
        if not self.concept_description:
            return None
        norm_query = self.normalizer.normalize(raw_attrs)
        best_sim = float('-inf')
        best_label = None
        for stored_raw, stored_label in self.concept_description:
            norm_stored = self.normalizer.normalize(stored_raw)
            sim = similarity(norm_query, norm_stored, self.is_numeric)
            if sim > best_sim:
                best_sim = sim
                best_label = stored_label
        return best_label

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def storage_count(self):
        """Number of instances currently in the concept description."""
        return len(self.concept_description)

    def storage_fraction(self, n_train):
        """Storage as a fraction of training set size."""
        return self.storage_count() / n_train if n_train > 0 else 0.0
