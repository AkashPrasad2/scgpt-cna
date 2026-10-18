# scGPT expression and data-confound investigation

Start with `03_expression_context_and_data.ipynb` in **Python (scGPT inference)**.
Keep every file in this folder together when uploading to Nibi. The main
notebook and `scgpt_session.py` must be in the same directory.

No package replacement, training, FlashAttention installation, or GPU is needed.
Use the existing scGPT 0.2.4 / torch 2.2.2+cpu environment. The scratch verification
environment used Python 3.12; the notebook is intended for the user's Python 3.10.

## The key audit correction

The official dev-temp `flash_layers.py` imports `MultiheadAttention` from its own
`layers.py`, **not** from `torch.nn`. That class applies identity projections
held as ordinary tensors rather than trainable parameters/buffers. Consequently
there are no cross-attention tensor entries in the checkpoint. Their absence
does not establish a checkpoint/source mismatch. Earlier discussion suggesting
otherwise was incorrect.

The source uses a shared Wqkv projection for known and unknown streams. The known
stream has a learned output projection. The unknown stream applies attention
with identity output projection and attends to known tokens plus only itself
among unknown tokens. It does **not** apply the learned known-stream output
projection. This distinction is preserved here.

## What the CPU implementation does

`AuditedSCGPT` uses the published scGPT gene/value embedding and decoder classes.
It translates the pinned official generative layers to native PyTorch scaled
dot-product attention, uses the unknown-gene flag, and loads every checkpoint
tensor with `strict=True`. For the provided architecture this is 163 tensors.
No unused tensors are silently discarded. It is inference-only, postnorm,
unbatched/unpadded per cell, without batch correction or explicit zero decoding.

The two outputs are:

- **GEP:** expression from each unknown gene's contextual transformer output.
- **MVC:** expression from the known-only CLS cell representation and each
  query's identity embedding.

Both use exactly the same known genes, targets, and target scale. The notebook
compares correct context with shuffled visible expression while holding gene
identities, target genes, value distribution, and scoring fixed. This shuffling
is an artificial sensitivity control, not a realistic biological condition.

## Verification and its limits

`test_scgpt_session.py` compares a small synthetic model with the **unmodified
official model/layer code** under `upstream/`. Only the FlashAttention known-
stream kernel is substituted with CPU SDPA for that reference run. The authors'
projection-free unknown-stream attention executes unchanged. Tests compare GEP,
MVC, and cell outputs, all 163 state-dict names, known/query separation,
no query-to-query leakage, analytical bin-tie moments, and the synthetic data
pipeline. Optional local reproduction needs `einops` for the official reference
source; the main notebook does not require it beyond existing scGPT dependencies.

These tests do **not** compare with a real GPU FlashAttention run, and do not
validate biological predictions. The user's actual checkpoint/data remain on
Nibi and must be tested there. Strict loading is a necessary compatibility check,
not proof of useful expression prediction.

## Fixed primary experiment

- One sample: C3N-00148-01.
- Full selection: 16 provisional CopyKAT diploid and 16 aneuploid cells.
- References: separate 64 cells from each group, with no overlap with evaluation.
- Original detected-gene eligibility: 1,000–3,000 genes per cell.
- Smoke run: first two selected cells from each group (four cells total).
- Up to 1,199 recognized expressed genes plus CLS; 15% expression targets;
  two mask repeats. This preserves the earlier experiment's settings. A 25%
  sensitivity analysis is a separate experiment, not silently substituted.
- Edges estimated from visible selected counts only. This is **not** an exact
  reproduction of full-cell binning used in pretraining. It avoids hidden-value
  influence on bin boundaries but introduces a documented distribution change.
- Tied-count target evaluation integrates over the full randomized bin support:
  expected MSE = squared error to the tie mean + irreducible tie variance.
- Complete reference profiles use full reference-cell bins, as in the previous
  baseline. This reference/evaluation scale approximation is documented.
- Baselines: pooled conditional-on-expression gene mean, CopyKAT-group-specific
  conditional gene mean, and constant. Group-specific means use provisional
  group information, so they are a stronger descriptive comparator, not an
  independently deployable malignancy classifier.
- Repeats/genes are averaged within cells before group comparisons. Thirty-two
  cells from one specimen are not thirty-two independent patient replicates.

## Data and cell identity

The notebook checks total counts, detected genes, mitochondrial count fraction,
and the fraction of expressed features with count one. All are descriptive;
none alone establishes malignancy or a universal exclusion threshold.

Small handpicked marker panels provide preliminary lineage/context evidence.
They are not calibrated annotation scores and are not comparable as probabilities
across panels of different genes. Kidney hypoxia markers can reflect a cell
state; they are not proof of cancer. Tumour cells can retain epithelial identity.
Biological cell-type confounding remains unresolved until reference mapping or
reviewed manual annotations are available. An optional explicit sample+barcode
join for Zaid's table is included; it does not claim those labels are infallible.

Explore error-QC and marker associations within CopyKAT groups. These are
exploratory correlations in a small, selected sample; no causal or patient-level
claims should be made. Sparse marker counts and different gene baselines can
create differences without CNAs.

The expressed-only target selection cannot test predicting true zeros or genes
lost completely to detection. Binning removes absolute expression scale and
does not directly measure DNA dosage or whole-genome doubling. No CNA calls,
WGS region comparisons, gene-coordinate joins, or tumour labels are invented.

## Reading the outputs

1. All 163 tensors load: structural compatibility.
2. Correct-vs-shuffled paired MSE < 0: correct expression associations helped.
3. Model-vs-gene-mean paired MSE < 0: advantage beyond that reference baseline.
4. Associations with counts/mitochondria/markers: possible confounds to examine,
   not proof that they caused errors.
5. Even a reconstruction advantage is not CNA detection. Regional/WGS evaluation
   needs verified genome assemblies, sample matching, ploidy interpretation,
   and appropriate cell populations.

Run the four-cell smoke test first. Then set `RUN_FULL = True` and execute the
benchmark onward to evaluate all 32 fixed cells. Keep settings, masks, reference
IDs and result tables; do not select a favourable run. Results and cell barcodes
are written locally on Nibi only when the notebook's save cell is run; the
provided notebook contains no biological data or executed outputs.

## Sources

- Official pinned pretraining model:
  https://github.com/bowang-lab/scGPT/blob/4068d67caaac1e28d56964da68e0214817e38428/scgpt/model/model.py
- Official generative attention:
  https://github.com/bowang-lab/scGPT/blob/4068d67caaac1e28d56964da68e0214817e38428/scgpt/model/flash_layers.py
- Official projection-free attention (critical audit finding):
  https://github.com/bowang-lab/scGPT/blob/4068d67caaac1e28d56964da68e0214817e38428/scgpt/model/layers.py
- Official training loop:
  https://github.com/bowang-lab/scGPT/blob/4068d67caaac1e28d56964da68e0214817e38428/examples/pretrain.py
- Published scGPT 0.2.4:
  https://pypi.org/project/scgpt/0.2.4/
- Official cell-reference mapping (a later route to cell-type labels without
  training a new classifier):
  https://scgpt.readthedocs.io/en/latest/tutorial_reference_mapping.html

Upstream source snapshots are supplied under their original MIT license.
