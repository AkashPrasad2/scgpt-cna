"""Synthetic verification, not validation on the Nibi checkpoint or biology.

Uses unmodified official dev-temp model/layer source. Only FlashAttention's
known-stream numerical kernel is substituted with torch SDPA on CPU. The
authors' custom unknown-stream MultiheadAttention executes unmodified.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import typing
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).parent
torch.set_num_threads(2)
torch.manual_seed(7)


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Extract only the upstream published embedding/decoder definitions, avoiding
# the unrelated scgpt package's import-time dependency stack in this test.
package("scgpt"); package("scgpt.model")
published = types.ModuleType("scgpt.model.model")
published.__dict__.update({k: v for k, v in vars(typing).items() if not k.startswith('__')})
published.__dict__.update(torch=torch, nn=nn, Tensor=torch.Tensor, np=np, F=F)
tree = ast.parse((ROOT / "upstream/published_0_2_4_model.py").read_text())
names = {"GeneEncoder", "ContinuousValueEncoder", "ExprDecoder", "MVCDecoder"}
tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
exec(compile(tree, "official_published_classes", "exec"), published.__dict__)
sys.modules[published.__name__] = published
tokenizer = types.ModuleType("scgpt.tokenizer")
tokenizer.GeneVocab = dict
sys.modules[tokenizer.__name__] = tokenizer

# Supply only the GPU kernel substitutes used by the original source.
package("flash_attn"); package("flash_attn.modules")
flash = types.ModuleType("flash_attn.flash_attention")


class CPUFlashAttention(nn.Module):
    def __init__(self, attention_dropout=0.0, **kwargs):
        super().__init__()

    def forward(self, qkv, key_padding_mask=None, need_weights=False, causal=False):
        assert key_padding_mask is None and not causal
        q, k, v = [qkv[:, :, j].transpose(1, 2) for j in range(3)]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return out.transpose(1, 2), None


flash.FlashAttention = CPUFlashAttention
flash.FlashMHA = CPUFlashAttention
sys.modules[flash.__name__] = flash
for name, attrs in {
    "flash_attn.flash_attn_interface": {"flash_attn_unpadded_qkvpacked_func": None},
    "flash_attn.bert_padding": {"unpad_input": None, "pad_input": None},
    "flash_attn.modules.mha": {"FlashCrossAttention": None},
}.items():
    mod = types.ModuleType(name); mod.__dict__.update(attrs); sys.modules[name] = mod

package("official"); package("official.model")
load("official.model.layers", ROOT / "upstream/layers.py")
official_layers = load("official.model.flash_layers", ROOT / "upstream/flash_layers.py")
for name, attr in [("official.model.dsbn", "DomainSpecificBatchNorm1d"),
                   ("official.model.grad_reverse", "grad_reverse")]:
    mod = types.ModuleType(name)
    setattr(mod, attr, None)
    sys.modules[name] = mod
official = load("official.model.model", ROOT / "upstream/model.py")
import scgpt_session as session

config = dict(embsize=32, nheads=4, d_hid=32, nlayers=12, dropout=0.0,
              input_style="binned", input_emb_style="continuous", n_bins=51,
              USE_GENERATIVE_TRAINING=True, MVC=True, no_cls=True, pad_token="<pad>",
              pad_value=-2)
vocab = {"<pad>": 0, "<cls>": 1, "<eoc>": 2,
         **{f"G{i}": i + 3 for i in range(1200)}}
ref = official.TransformerModel(ntoken=len(vocab), d_model=32, nhead=4,
    d_hid=32, nlayers=12, n_cls=1, vocab=vocab, dropout=0.0,
    pad_token="<pad>", pad_value=-2, do_mvc=True,
    use_generative_training=True, use_fast_transformer=True).eval()
new = session.AuditedSCGPT(config, vocab).eval()
assert len(ref.state_dict()) == len(new.state_dict()) == 163
assert set(ref.state_dict()) == set(new.state_dict())
new.load_state_dict(ref.state_dict(), strict=True)

known = torch.tensor([[1, 3, 4, 5, 6], [1, 7, 8, 9, 10]])
values = torch.tensor([[-2., 1., 8., 20., 45.], [-2., 5., 10., 30., 50.]])
query = torch.tensor([[11, 12, 13], [14, 15, 16]])
with torch.inference_mode():
    expected = ref(known, values, torch.zeros_like(known, dtype=torch.bool), query,
        torch.zeros_like(query, dtype=torch.bool), MVC=True, generative_training=True)
    actual = new(known, values, query)
    for key, old_key in [("GEP", "gen_preds"), ("MVC", "mvc_output"), ("cell_emb", "cell_emb")]:
        e = expected[old_key][:, known.shape[1]:] if old_key == "mvc_output" else expected[old_key]
        torch.testing.assert_close(actual[key], e, rtol=2e-5, atol=2e-5)
        print(key, "max error vs official:", float((actual[key] - e).abs().max()))
    altered_query = query.clone(); altered_query[:, 1] += 10
    altered = new(known, values, altered_query)
    torch.testing.assert_close(actual["cell_emb"], altered["cell_emb"], rtol=0, atol=0)
    torch.testing.assert_close(actual["GEP"][:, 0], altered["GEP"][:, 0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual["MVC"][:, 0], altered["MVC"][:, 0], rtol=0, atol=0)
print("PASS: 163 tensor names match; strict load; known/query separation and no query-to-query leakage.")

# Tie moments agree with direct enumeration, including out-of-range values.
edges = np.array([1., 1., 1., 2., 2., 4.])
x = np.array([.5, 1., 1.5, 2., 4., 5.])
m, v = session.tie_moments(x, edges, n_bins=7)
for j, value in enumerate(x):
    left = np.digitize(value, edges); right = np.digitize(value, edges, right=True)
    support = np.arange(right + 1, left + 1) if left > right else np.array([left])
    support = np.clip(support, 1, 6)
    np.testing.assert_allclose(m[j], support.mean())
    np.testing.assert_allclose(v[j], support.var())
print("PASS: tie moments, repeated boundaries, and outside-boundary clipping.")

# Exercise full data/QC/reference/sampling/scoring flow on synthetic counts.
import pandas as pd
from scipy import sparse
symbols = [f"G{i}" for i in range(1200)]
obs = pd.DataFrame({"copykat_prediction": ["diploid"] * 10 + ["aneuploid"] * 10},
                   index=pd.Index([f"cell{i}" for i in range(20)], name="barcode"))
counts = sparse.csr_matrix(np.random.default_rng(1).integers(1, 8, (20, 1200)))
adata = types.SimpleNamespace(layers={"counts": counts}, obs=obs,
    var=pd.DataFrame({"gene_id": [f"ID{i}" for i in range(1200)], "gene_symbol": symbols}))
qc, _ = session.qc_and_markers(adata)
settings = dict(seed=42, eval_per_group=2, ref_per_group=2,
                max_genes=30, mask_fraction=.25, repeats=2)
evaluation, reference = session.select_cells(qc, settings)
baselines = session.reference_baselines(counts, qc, reference, 42)
result = session.benchmark(new, vocab, counts, np.arange(3, 1203), adata.var,
                           qc, evaluation, baselines, settings, progress=False)
assert len(result["scores"]) == 4 * 2 * 7
assert len(result["per_cell"]) == 4 * 7
assert result["per_cell"]["total_counts"].notna().all()
assert np.isfinite(result["scores"]["expected_MSE"]).all()
paired = session.paired_context_results(result["per_cell"])
assert len(paired) == 4
session.qc_associations(result["per_cell"])
print("PASS: end-to-end synthetic QC, separate references, masks, both heads, context control, summaries.")
