# Real-data verification — not simulated

Aura-State's guarantees on real public datasets, so the numbers are real.

| Example | What it shows |
|---|---|
| [`conformal_on_real_data.py`](conformal_on_real_data.py) | Aura-State's conformal interval on scikit-learn's **diabetes** dataset (442 real patient records). Requested 90% coverage → **91.3% real empirical coverage** on held-out patients. |

```bash
pip install scikit-learn
python examples/real_data/conformal_on_real_data.py
```

The verification layer (Z3 / CTL / taint / conformal) needs no LLM and no key —
it runs on real data directly.
