"""
Data loader for the two datasets used in this project.

Expected file locations (place cleaned CSVs here):
  data/voting.csv
  data/waveform.csv

--- Voting (Congressional Voting Records, UCI) ---
  435 instances, 16 Boolean attributes + 1 class label
  Attributes: 16 yes/no/missing votes
  Class: democrat / republican
  Missing values represented as '?' in the original; we treat as MISSING.
  Paper split: 350 train / 85 test

  Expected CSV format (no header, or header row will be auto-detected):
    class, vote1, vote2, ..., vote16
  where class is the first column.

--- Waveform (Breiman et al. 1984, UCI) ---
  Generated dataset; 800 instances used here (300 train / 500 test).
  21 continuous attributes, 3 classes (0, 1, 2).
  No missing values.

  Expected CSV format:
    attr1, attr2, ..., attr21, class
  where class is the LAST column.

Both loaders return:
  X_train, y_train, X_test, y_test, is_numeric_flags
"""

import csv
import os
import random

MISSING = None


def _split(data, train_size, test_size, shuffle=True, seed=None):
    """
    Split data into train/test of exactly the specified sizes.
    Raises if len(data) < train_size + test_size.
    """
    if len(data) < train_size + test_size:
        raise ValueError(
            f"Dataset has {len(data)} instances but "
            f"train_size + test_size = {train_size + test_size}"
        )
    if shuffle:
        rng = random.Random(seed)
        data = list(data)
        rng.shuffle(data)
    train = data[:train_size]
    test = data[train_size: train_size + test_size]
    return train, test


def _unzip(data):
    """Split list of (attrs, label) pairs into X, y lists."""
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    return X, y


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

VOTING_TRAIN_SIZE = 350
VOTING_TEST_SIZE = 85
# All 16 attributes are Boolean/symbolic (yes, no, ?)
VOTING_IS_NUMERIC = [False] * 16

# Map raw vote strings to canonical values
_VOTE_MAP = {'y': 'y', 'yes': 'y', 'n': 'n', 'no': 'n', '?': MISSING}


def load_voting(data_dir='data', seed=None):
    """
    Load and split the Congressional Voting dataset.

    The CSV should have 17 columns: class first, then 16 vote attributes.
    Class values: 'democrat' / 'republican' (or 'd' / 'r' — auto-detected).

    Returns
    -------
    X_train, y_train, X_test, y_test, is_numeric_flags
    """
    path = os.path.join(data_dir, 'voting.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Voting dataset not found at '{path}'. "
            "Place your cleaned CSV there."
        )

    with open(path, newline='') as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    # Skip header if present (heuristic: first column of first row is not
    # a recognized class label or vote value)
    start = 0
    first = all_rows[0]
    if first[0].lower() not in ('democrat', 'republican', 'd', 'r', '1', '2',
                                'dem', 'rep'):
        start = 1  # skip header

    data = []
    for row in all_rows[start:]:
        if len(row) < 17:
            continue  # skip malformed rows
        label = row[0].strip().lower()
        attrs = []
        for val in row[1:17]:
            v = val.strip().lower()
            attrs.append(_VOTE_MAP.get(v, v))
        data.append((attrs, label))

    train_data, test_data = _split(
        data, VOTING_TRAIN_SIZE, VOTING_TEST_SIZE, seed=seed
    )
    X_train, y_train = _unzip(train_data)
    X_test, y_test = _unzip(test_data)
    return X_train, y_train, X_test, y_test, VOTING_IS_NUMERIC


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

WAVEFORM_TRAIN_SIZE = 300
WAVEFORM_TEST_SIZE = 500
WAVEFORM_IS_NUMERIC = [True] * 21


def load_waveform(data_dir='data', seed=None):
    """
    Load and split the Waveform dataset.

    The CSV should have 22 columns: 21 numeric attributes + class label (last).
    Class values: 0, 1, 2 (integer or string).

    Returns
    -------
    X_train, y_train, X_test, y_test, is_numeric_flags
    """
    path = os.path.join(data_dir, 'waveform.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Waveform dataset not found at '{path}'. "
            "Place your cleaned CSV there."
        )

    with open(path, newline='') as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    # Skip header if first row is non-numeric in attribute columns
    start = 0
    if all_rows:
        first = all_rows[0]
        try:
            float(first[0])
        except ValueError:
            start = 1

    data = []
    for row in all_rows[start:]:
        if len(row) < 22:
            continue
        try:
            attrs = [float(v.strip()) for v in row[:21]]
        except ValueError:
            continue
        label = row[21].strip()
        data.append((attrs, label))

    train_data, test_data = _split(
        data, WAVEFORM_TRAIN_SIZE, WAVEFORM_TEST_SIZE, seed=seed
    )
    X_train, y_train = _unzip(train_data)
    X_test, y_test = _unzip(test_data)
    return X_train, y_train, X_test, y_test, WAVEFORM_IS_NUMERIC
