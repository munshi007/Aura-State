# Real-data verification — not simulated

Aura-State's guarantees on real public datasets, so the numbers are real. No LLM,
no API key — the verification layer (Z3 / conformal) runs on real data directly.

| Example | What it shows | Real result |
|---|---|---|
| [`verify_real_dataset.py`](verify_real_dataset.py) | Aura-State's **Z3** proves the arithmetic invariants of **1,000 real sales records** (`revenue = units × price`, `cost = units × unit_cost`, `profit = revenue − cost`), then shows it rejecting a corrupted record. | 1,000/1,000 verified · 3,000 obligations · ~1,200 records/sec |
| [`conformal_on_real_data.py`](conformal_on_real_data.py) | Aura-State's **conformal** interval on scikit-learn's **diabetes** dataset (442 real patient records). | requested 90% → **91.3%** real coverage on held-out patients |

```bash
python examples/real_data/verify_real_dataset.py        # no deps beyond aura-state
pip install scikit-learn
python examples/real_data/conformal_on_real_data.py
```

Datasets are real and public — see [`SOURCE.txt`](SOURCE.txt) for attribution.
