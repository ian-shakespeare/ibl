# Instance-Based Learning Algorithms (IB1, IB2, IB3)

[Slide deck presentation available here.](https://github.com/ian-shakespeare/ibl/blob/main/presentation.pdf)

Implementation of Aha, Kibler & Albert (1991),  
*"Instance-Based Learning Algorithms"*, Machine Learning, 6, 37–66.

**Canonical base model:** k-Nearest Neighbors  
**Paper's contribution:** IB2 (storage reduction) and IB3 (noise-tolerant storage reduction)

---

## What this implements

| Algorithm | Description |
|-----------|-------------|
| **IB1** | Baseline: saves every instance, classifies by 1-nearest-neighbor |
| **IB2** | Saves only misclassified instances — reduces storage, sensitive to noise |
| **IB3** | Extends IB2 with a confidence-interval filter that identifies and discards noisy instances |

The three claims tested (Table 6 of the paper):
- **(a)** IB2 reduces storage substantially at a small accuracy cost on clean data  
- **(b)** IB3 handles noise better than IB2 (higher accuracy on noisy domains)  
- **(c)** IB3 still significantly reduces storage compared to IB1

---

## Repository structure

```
ibl/
├── src/
│   ├── similarity.py     # Normalized Euclidean similarity function + running normalizer
│   ├── ib1.py            # IB1 algorithm
│   ├── ib2.py            # IB2 algorithm
│   ├── ib3.py            # IB3 algorithm (main contribution)
│   └── data_loader.py    # Dataset loaders for Voting and Waveform
├── tests/
│   └── test_ibl.py       # Unit tests (28 tests, no dependencies)
├── data/                 # ← place your CSV files here (see below)
├── results/              # Experiment output written here
├── experiment.py         # Main experiment script
└── requirements.txt
```

---

## Requirements

- Python 3.7 or later  

To run tests with pytest (optional):
```bash
pip install pytest
```

---

## Data preparation

Place two cleaned CSV files in the `data/` directory:

### `data/voting.csv`
- Source: [UCI Congressional Voting Records](https://archive.ics.uci.edu/ml/datasets/congressional+voting+records)
- 435 rows, 17 columns
- **Format:** `class, vote1, vote2, ..., vote16`  
  - Class column first: `democrat` or `republican`  
  - Vote values: `y`, `n`, or `?` (missing)
- Header row is optional (auto-detected)

### `data/waveform.csv`
- Source: [UCI Waveform Database Generator](https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+%28Version+1%29)
- 800+ rows, 22 columns
- **Format:** `attr1, attr2, ..., attr21, class`  
  - 21 continuous numeric attributes, then class label (0, 1, or 2) last
- Header row is optional (auto-detected)

---

## Running the experiments

### Full replication (50 trials, matches Table 6)

```bash
python experiment.py
```

This runs 50 independent trials per dataset, each with a fresh random
train/test split using the paper's exact split sizes:
- Voting: 350 train / 85 test
- Waveform: 300 train / 500 test

Results are printed to stdout and saved to `results/results.txt`.

### Custom options

```bash
python experiment.py --trials 10          # faster run for testing
python experiment.py --data-dir /path/to/data
python experiment.py --seed 123           # reproducible shuffles
```

### Running unit tests

```bash
python tests/test_ibl.py
# or, with pytest:
pytest tests/ -v
```

---

## Expected output (approximate)

Results will closely match Table 6 from the paper:

```
Dataset        IB1 Acc    IB1 Stor    IB2 Acc    IB2 Stor    IB3 Acc    IB3 Stor
---------------------------------------------------------------------------
Voting       91.8 ±0.4    100.0%    90.9 ±0.5     11.1%    91.6 ±0.5      7.4%
Waveform     75.2 ±0.3    100.0%    69.6 ±0.4     32.5%    73.8 ±0.4     14.6%
```

Minor differences from the paper are expected due to:
- Different random seeds and shuffling
- Wilson score CIs vs. the paper's normal approximation intervals
- Floating-point differences in normalization

---

## Implementation notes

**Normalization:** Raw attribute values are stored in the concept description.
Normalization to [0,1] using running min/max is applied at *query time*, not
storage time. This is necessary because the normalization range can expand as
new extreme values arrive during incremental training.

**IB3 significance test:** Uses Wilson score confidence intervals rather than
the normal approximation described in the paper (Hogg & Tanis 1983, Eq. 5.5-4).
Wilson intervals are more accurate for small sample sizes and extreme proportions.
Acceptance threshold: z = 90%. Dropping threshold: z = 75% (matches paper defaults).

**Missing values (Voting dataset):** `?` entries are treated as maximally
dissimilar from any present value (f = 1.0), per the paper's definition.
