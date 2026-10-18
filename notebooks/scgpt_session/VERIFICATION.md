# Verification completed

The local synthetic verification script passed with PyTorch 2.2.2+cpu and
NumPy 1.26.4 under Python 3.12. It does not use the Nibi checkpoint or dataset.

Reference comparison: unmodified official source at commit
`4068d67caaac1e28d56964da68e0214817e38428`, with only the known-stream
FlashAttention kernel replaced by CPU scaled dot-product attention. The
authors' custom unknown-stream attention executes unchanged.

Synthetic model: 12 layers, width 32, four heads, dropout disabled for evaluation.
Both implementations have exactly the same 163 state-dict keys and load strictly.

Maximum absolute output differences:

| Output | Difference |
| --- | ---: |
| Generative expression prediction | 2.98e-08 |
| MVC expression prediction | 4.17e-07 |
| Cell embedding | 0 |

The following also passed:

- Changing a queried gene leaves the known-cell representation unchanged.
- Changing one queried gene leaves other query predictions unchanged.
- Analytical target moments match enumeration of randomized bin-tie support.
- A synthetic sparse-count benchmark exercises QC summaries, separate reference
  cells, masks, both prediction heads, shuffled-context controls, baselines,
  cell-level summaries, and QC association calculations.
- All notebook code cells parse with Python 3.10 syntax.

These checks establish numerical agreement for the tested inference computation.
They do not establish GPU-kernel equivalence for every input, useful biological
prediction, correct cell typing, CNA detection, or successful loading of the
user's checkpoint. The four-cell Nibi run is the next check.

Optional reproduction from this folder, in an environment with the notebook's
dependencies and `einops`:

```bash
python test_scgpt_session.py
```

The script supplies temporary Python import stubs within its own process to
isolate the reference source. It does not modify the installed scGPT package.
