"""
IB2 Algorithm — Aha, Kibler & Albert (1991), Table 2.

Identical to IB1 except it saves only misclassified instances.
This dramatically reduces storage requirements but is sensitive to noise.

CD <- {}
for each x in TrainingSet:
    1. compute Similarity(x, y) for each y in CD
    2. ymax <- y in CD with maximal Similarity
    3. if class(x) == class(ymax):
           classification <- correct
       else:
           classification <- incorrect
           CD <- CD ∪ {x}       # save ONLY if misclassified

Raw attribute values are stored; normalization is applied at query time
using the running min/max, so re-scaling as new extremes arrive is correct.
"""

from .similarity import similarity, AttributeNormalizer


class IB2:
    """
    IB2: storage-reducing IBL algorithm.
    Saves only instances that were misclassified at the time of training.

    Parameters
    ----------
    is_numeric_flags : list of bool
        One entry per predictor attribute; True means numeric.
    """

    def __init__(self, is_numeric_flags):
        self.is_numeric = is_numeric_flags
        self.normalizer = AttributeNormalizer(is_numeric_flags)
        # Each entry: (raw_attributes, class_label)
        self.concept_description = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Incrementally train on a sequence of instances.

        Returns
        -------
        train_accuracy : float
        """
        correct = 0
        for attrs, label in zip(X, y):
            self.normalizer.update(attrs)

            if self.concept_description:
                pred = self._classify(attrs)
                if pred == label:
                    correct += 1
                else:
                    # Save only misclassified instances (raw values)
                    self.concept_description.append((attrs, label))
            else:
                # CD is empty — save first instance unconditionally
                self.concept_description.append((attrs, label))

        n = len(X)
        train_accuracy = correct / n if n > 0 else 0.0
        return train_accuracy

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        predictions = []
        for attrs in X:
            pred = self._classify(attrs)
            predictions.append(pred)
        return predictions

    def _classify(self, raw_attrs):
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
        return len(self.concept_description)

    def storage_fraction(self, n_train):
        return self.storage_count() / n_train if n_train > 0 else 0.0
