"""scGPT checkpoint audit and paired expression/context experiments.

Inference-only CPU translation of the authors' dev-temp generative computation.
Source: bowang-lab/scGPT commit 4068d67caaac1e28d56964da68e0214817e38428
See README.md and upstream LICENSE. No training, padding, or batch correction.
"""
from pathlib import Path
import json
import math
import hashlib
import importlib.metadata

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
import torch
from torch import nn
from torch.nn import functional as F
from scgpt.model.model import (
    GeneEncoder, ContinuousValueEncoder, ExprDecoder, MVCDecoder,
)
from scgpt.tokenizer import GeneVocab

UPSTREAM_COMMIT = "4068d67caaac1e28d56964da68e0214817e38428"


class GenerativeAttention(nn.Module):
    """Shared QKV; known output projection; projection-free unknown attention.

    The absence of out_proj on the UNKNOWN stream is intentional: upstream's
    custom MultiheadAttention uses fixed identity input/output projections.
    PyTorch SDPA bool masks use True=allowed, unlike MultiheadAttention masks.
    """
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.Wqkv = nn.Linear(width, 3 * width)
        self.out_proj = nn.Linear(width, width)

    def project(self, x):
        b, n, d = x.shape
        qkv = self.Wqkv(x).reshape(b, n, 3, self.heads, d // self.heads)
        return tuple(qkv[:, :, i].transpose(1, 2) for i in range(3))

    @staticmethod
    def merge(x):
        return x.transpose(1, 2).contiguous().flatten(2)

    def forward(self, known, unknown):
        qk, kk, vk = self.project(known)
        known_out = self.out_proj(self.merge(F.scaled_dot_product_attention(
            qk, kk, vk, dropout_p=0.0,
        )))
        if unknown is None:
            return known_out, None
        qu, ku, vu = self.project(unknown)
        nq, nk = unknown.shape[1], known.shape[1]
        allowed = torch.cat([
            torch.ones(nq, nk, dtype=torch.bool, device=known.device),
            torch.eye(nq, dtype=torch.bool, device=known.device),
        ], dim=1)
        unknown_out = self.merge(F.scaled_dot_product_attention(
            qu, torch.cat([kk, ku], dim=2), torch.cat([vk, vu], dim=2),
            attn_mask=allowed, dropout_p=0.0,
        ))
        return known_out, unknown_out


class GenerativeLayer(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.self_attn = GenerativeAttention(width, heads)
        self.linear1 = nn.Linear(width, hidden)
        self.linear2 = nn.Linear(hidden, width)
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)

    def finish(self, x, attention):
        x = self.norm1(x + attention)
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))

    def forward(self, known, unknown):
        ak, au = self.self_attn(known, unknown)
        known = self.finish(known, ak)
        unknown = None if unknown is None else self.finish(unknown, au)
        return known, unknown


class GenerativeStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            GenerativeLayer(config["embsize"], config["nheads"], config["d_hid"])
            for _ in range(config["nlayers"])
        ])

    def forward(self, known, unknown=None):
        for layer in self.layers:
            known, unknown = layer(known, unknown)
        return known, unknown


class AuditedSCGPT(nn.Module):
    """Checkpoint-shaped model for the supplied binned, continuous, postnorm model.

    Uses upstream embedding/decoder classes and reproduces the dev-temp
    transformer_generate path. Processes one unpadded sequence per call.
    No optimizer or training support. All checkpoint tensors must load strictly.
    """
    def __init__(self, config, vocab):
        super().__init__()
        assert config["input_style"] == "binned"
        assert config["input_emb_style"] == "continuous"
        assert config.get("USE_GENERATIVE_TRAINING") is True
        assert config.get("MVC") is True
        assert not config.get("pre_norm", False)
        assert not config.get("domain_spec_batchnorm", False)
        assert config.get("no_cls", True)
        width = config["embsize"]
        self.config = config
        self.encoder = GeneEncoder(len(vocab), width, vocab[config["pad_token"]])
        self.value_encoder = ContinuousValueEncoder(width, config["dropout"])
        self.flag_encoder = nn.Embedding(2, width)
        self.transformer_encoder = GenerativeStack(config)
        self.decoder = ExprDecoder(width, explicit_zero_prob=False)
        self.mvc_decoder = MVCDecoder(width, arch_style="inner product",
                                      explicit_zero_prob=False)

    def forward(self, known_ids, known_values, query_ids):
        if self.training:
            raise RuntimeError("Inference only: call model.eval() first.")
        assert known_ids.ndim == 2 and known_ids.shape == known_values.shape
        assert query_ids.ndim == 2 and known_ids.shape[0] == query_ids.shape[0]
        assert known_ids.shape[1] >= 1 and query_ids.shape[1] >= 1
        assert torch.all(known_values[:, 1:] > 0), "Only expressed context is supported."
        known_emb = self.encoder(known_ids) + self.value_encoder(known_values)
        query_emb = self.encoder(query_ids)
        unknown_emb = query_emb + self.flag_encoder.weight[1].view(1, 1, -1)
        known, unknown = self.transformer_encoder(known_emb, unknown_emb)
        cell = known[:, 0, :]
        return {
            "GEP": self.decoder(unknown)["pred"],
            "MVC": self.mvc_decoder(cell, query_emb)["pred"],
            "cell_emb": cell,
        }


def load_model(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "args.json").read_text())
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    assert all(t in vocab for t in ["<cls>", "<pad>", "<eoc>"])
    model = AuditedSCGPT(config, vocab)
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    for wrapper in ["state_dict", "model_state_dict"]:
        if wrapper in state:
            state = state[wrapper]
            break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    print(f"Strict load: {len(state)} checkpoint tensors; no omitted tensors.")
    return model, vocab, config


def load_sample(data_dir, copykat_path):
    import anndata as ad
    data_dir = Path(data_dir)
    features = pd.read_csv(data_dir / "features.tsv", sep="\t", header=None, dtype=str)
    barcodes = pd.read_csv(data_dir / "barcodes.tsv", sep="\t", header=None, dtype=str)
    matrix = mmread(data_dir / "matrix.mtx").T.tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    assert matrix.shape == (len(barcodes), len(features))
    assert (matrix.data >= 0).all() and np.allclose(matrix.data, np.round(matrix.data))
    assert features.shape[1] >= 3 and features[2].eq("Gene Expression").all()
    var = pd.DataFrame({"gene_id": features[0].to_numpy(),
                        "gene_symbol": features[1].to_numpy()})
    var.index = pd.Index(var["gene_id"].to_numpy(), name="feature_id")
    assert var.index.is_unique, "Resolve duplicate feature IDs before proceeding."
    obs = pd.DataFrame(index=pd.Index(barcodes[0].to_numpy(), name="barcode"))
    assert obs.index.is_unique
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    ck = pd.read_csv(copykat_path, sep="\t", dtype=str)
    assert not ck["cell.names"].duplicated().any()
    adata.obs["copykat_prediction"] = ck.set_index("cell.names")["copykat.pred"].reindex(obs.index)
    return adata


MARKERS = {
    "T_cell": ["CD3D", "CD3E", "TRAC"],
    "B_cell": ["MS4A1", "CD79A", "CD79B"],
    "myeloid": ["LST1", "TYROBP", "FCER1G", "LYZ"],
    "endothelial": ["PECAM1", "VWF", "EMCN"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19"],
    "proximal_tubule": ["LRP2", "CUBN", "SLC34A1"],
    "renal_hypoxia": ["CA9", "NDUFA4L2"],
}


def qc_and_markers(adata):
    """Descriptive marker evidence; NOT calibrated scores or assigned cell types."""
    x = sparse.csr_matrix(adata.layers["counts"])
    symbols = adata.var["gene_symbol"].astype(str).to_numpy()
    total = np.asarray(x.sum(axis=1)).ravel()
    detected = np.asarray((x > 0).sum(axis=1)).ravel()
    mt = np.char.startswith(symbols.astype(str), "MT-")
    mt_counts = np.asarray(x[:, mt].sum(axis=1)).ravel()
    obs = adata.obs.copy()
    obs["total_counts"] = total
    obs["n_genes_detected"] = detected
    obs["pct_counts_mt"] = 100 * mt_counts / np.maximum(total, 1)
    positive = x.copy()
    positive.data = (positive.data == 1).astype(np.float32)
    obs["fraction_detected_genes_count1"] = np.asarray(positive.sum(axis=1)).ravel() / np.maximum(detected, 1)
    scale = 1e4 / np.maximum(total, 1)
    coverage = []
    for name, genes in MARKERS.items():
        idx = np.flatnonzero(np.isin(symbols, genes))
        present = sorted(set(symbols[idx]))
        coverage.append({"panel": name, "present": ", ".join(present),
                         "n_present_symbols": len(present), "n_requested": len(genes)})
        if len(idx):
            small = x[:, idx].toarray().astype(float) * scale[:, None]
            obs["marker_" + name] = np.log1p(small).mean(axis=1)
        else:
            obs["marker_" + name] = np.nan
    return obs, pd.DataFrame(coverage)


def prepare_expression(adata, vocab):
    symbols = adata.var["gene_symbol"]
    keep = symbols.isin(vocab.get_stoi()) & ~symbols.duplicated()
    columns = np.flatnonzero(keep.to_numpy())
    symbols = symbols.iloc[columns].to_numpy()
    ids = np.array([vocab[s] for s in symbols], dtype=np.int64)
    x = sparse.csr_matrix(adata.layers["counts"][:, columns], dtype=np.float32)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    gene_info = adata.var.iloc[columns].copy().reset_index(drop=True)
    print(f"Model features: {len(columns):,}; first occurrence retained for duplicate symbols.")
    return x, ids, gene_info


def select_cells(obs, settings):
    rng = np.random.default_rng(settings["seed"])
    evaluation, reference = [], []
    for group in ["diploid", "aneuploid"]:
        eligible = np.flatnonzero(
            obs["copykat_prediction"].eq(group).to_numpy()
            & obs["n_genes_detected"].between(1000, 3000).to_numpy())
        n = settings["eval_per_group"] + settings["ref_per_group"]
        assert len(eligible) >= n, f"Only {len(eligible)} eligible {group} cells."
        choice = rng.choice(eligible, n, replace=False)
        evaluation.extend(choice[:settings["eval_per_group"]])
        reference.extend(choice[settings["eval_per_group"]:])
    assert not set(evaluation) & set(reference)
    return np.array(evaluation), np.array(reference)


def positive_row(x, i):
    row = x.getrow(int(i))
    return row.indices.copy(), row.data.copy()


def tie_moments(values, edges, n_bins=51):
    left = np.digitize(values, edges, right=False)
    right = np.digitize(values, edges, right=True)
    lo = np.clip(np.where(left > right, right + 1, left), 1, n_bins - 1)
    hi = np.clip(left, 1, n_bins - 1)
    assert np.all(lo <= hi)
    mean = (lo + hi) / 2
    variance = ((hi - lo + 1) ** 2 - 1) / 12
    return mean, variance


def randomized_bins(values, edges, seed, n_bins=51):
    # Reproduce scGPT _digitize without mutating NumPy's global random state.
    left = np.digitize(values, edges)
    right = np.digitize(values, edges, right=True)
    u = np.random.RandomState(seed).rand(len(values))
    return np.clip(np.ceil(u * (right - left) + left), 1, n_bins - 1).astype(np.float32)


def reference_baselines(x, obs, reference, seed, n_bins=51):
    """Legacy-compatible bin averaging, conditional on expression > 0.

    Edges use all recognized expressed genes of each separate reference cell.
    Unlike evaluation's observed-only bins, these reference profiles are complete;
    this approximation is explicit and is kept fixed across tested model routes.
    """
    output = {}
    for label in ["pooled", "diploid", "aneuploid"]:
        sums = np.zeros(x.shape[1]); n = np.zeros(x.shape[1], dtype=np.int64)
        for i in reference:
            if label != "pooled" and obs.iloc[i]["copykat_prediction"] != label:
                continue
            cols, values = positive_row(x, i)
            edges = np.quantile(values, np.linspace(0, 1, n_bins - 1))
            b = randomized_bins(values, edges, seed + int(i), n_bins)
            sums[cols] += b; n[cols] += 1
        constant = sums.sum() / max(n.sum(), 1)
        means = np.full(x.shape[1], constant)
        np.divide(sums, n, out=means, where=n > 0)
        output[label] = {"mean": means, "support": n, "constant": constant}
    return output


def benchmark(model, vocab, x, gene_ids, gene_info, obs, evaluation, baselines,
              settings, progress=True):
    """Paired GEP/MVC context test, fixed targets and masks for both routes."""
    rows, targets, cases = [], [], []
    n_bins = int(model.config["n_bins"])
    pad = float(model.config["pad_value"])
    for number, i in enumerate(evaluation, 1):
        all_cols, all_values = positive_row(x, i)
        rng = np.random.default_rng(settings["seed"] + int(i))
        pick = np.sort(rng.choice(len(all_cols), min(settings["max_genes"], len(all_cols)), replace=False))
        cols, values = all_cols[pick], all_values[pick]
        group = obs.iloc[i]["copykat_prediction"]
        for repeat in range(settings["repeats"]):
            seed = settings["seed"] + 100000 * repeat + int(i)
            rng = np.random.default_rng(seed)
            nh = max(1, int(round(settings["mask_fraction"] * len(cols))))
            hidden = np.sort(rng.choice(len(cols), nh, replace=False))
            visible = np.ones(len(cols), dtype=bool); visible[hidden] = False
            edges = np.quantile(values[visible], np.linspace(0, 1, n_bins - 1))
            b = randomized_bins(values[visible], edges, seed, n_bins)
            mean, variance = tie_moments(values[hidden], edges, n_bins)
            known_ids = torch.tensor([[vocab["<cls>"], *gene_ids[cols[visible]].tolist()]])
            query_ids = torch.tensor(gene_ids[cols[hidden]]).unsqueeze(0)
            metadata = {"cell": obs.index[i], "copykat_group": group, "repeat": repeat}
            for j, col in enumerate(cols[hidden]):
                targets.append({**metadata, "gene_symbol": gene_info.iloc[col]["gene_symbol"],
                                "raw_count": values[hidden][j], "tie_mean": mean[j],
                                "tie_variance": variance[j]})
            results = {}
            for condition in ["correct", "shuffled"]:
                bv = b if condition == "correct" else np.random.default_rng(seed + 9999).permutation(b)
                known_values = torch.tensor([[pad, *bv.tolist()]], dtype=torch.float32)
                with torch.inference_mode():
                    out = model(known_ids, known_values, query_ids)
                for route in ["MVC", "GEP"]:
                    p = out[route][0].cpu().numpy()
                    assert np.isfinite(p).all()
                    results[route + "_" + condition] = p
            pooled = baselines["pooled"]
            results["gene_mean"] = pooled["mean"][cols[hidden]]
            results["group_gene_mean"] = baselines[group]["mean"][cols[hidden]]
            results["constant"] = np.full(nh, pooled["constant"])
            for method, p in results.items():
                rows.append({**metadata, "method": method, "n_targets": nh,
                             "expected_MSE": np.mean((p - mean)**2 + variance),
                             "tie_mean_MSE": np.mean((p - mean)**2),
                             "prediction_std": np.std(p), "prediction_mean": np.mean(p),
                             "mean_tie_variance": np.mean(variance),
                             "baseline_fallback_fraction": np.mean(pooled["support"][cols[hidden]] == 0)})
            cases.append(pd.DataFrame({**metadata,
                "gene_id": gene_info.iloc[cols[hidden]]["gene_id"].to_numpy(),
                "gene_symbol": gene_info.iloc[cols[hidden]]["gene_symbol"].to_numpy(),
                "target_tie_mean": mean, "target_tie_variance": variance,
                **{k: v for k, v in results.items() if k.startswith(("GEP", "MVC"))}}))
        if progress:
            print(f"Completed {number}/{len(evaluation)} cells", flush=True)
    scores = pd.DataFrame(rows)
    per_cell = scores.groupby(["cell", "copykat_group", "method"], as_index=False).mean(numeric_only=True)
    per_cell = per_cell.merge(obs.reset_index().rename(columns={obs.index.name or "index": "cell"}),
                              on=["cell", "copykat_group"] if "copykat_group" in obs.columns else "cell",
                              how="left", validate="many_to_one")
    summary = per_cell.groupby(["copykat_group", "method"]).agg(
        n_cells=("cell", "nunique"), mean_expected_MSE=("expected_MSE", "mean"),
        median_expected_MSE=("expected_MSE", "median"), mean_prediction_std=("prediction_std", "mean"))
    return {"scores": scores, "per_cell": per_cell, "summary": summary,
            "targets": pd.DataFrame(targets), "predictions": pd.concat(cases, ignore_index=True)}


def paired_context_results(per_cell):
    wide = per_cell.pivot(index=["cell", "copykat_group"], columns="method", values="expected_MSE")
    for route in ["MVC", "GEP"]:
        # Negative = correct context helped.
        wide[route + "_correct_minus_shuffled"] = wide[route + "_correct"] - wide[route + "_shuffled"]
        wide[route + "_minus_gene_mean"] = wide[route + "_correct"] - wide["gene_mean"]
    return wide


def qc_associations(per_cell):
    features = ["total_counts", "n_genes_detected", "pct_counts_mt",
                "fraction_detected_genes_count1"] + [c for c in per_cell if c.startswith("marker_")]
    rows = []
    for (group, method), frame in per_cell.groupby(["copykat_group", "method"]):
        if method not in ["MVC_correct", "GEP_correct", "gene_mean"]:
            continue
        for feature in features:
            pair = frame[[feature, "expected_MSE"]].dropna()
            rho = pair.corr(method="spearman").iloc[0, 1] if len(pair) >= 4 else np.nan
            rows.append({"group": group, "method": method, "feature": feature,
                         "n_cells": len(pair), "spearman_rho": rho})
    return pd.DataFrame(rows)


def load_manual_labels(obs, path, sample_id, sample_col="sample_id",
                       barcode_col="barcode"):
    """Optional explicit-key join. Labels remain reference annotations, not truth."""
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        labels = pd.read_excel(path, dtype=str)
    else:
        labels = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", dtype=str)
    assert sample_col in labels and barcode_col in labels
    labels = labels.loc[labels[sample_col].eq(sample_id)].copy()
    assert not labels[barcode_col].duplicated().any()
    labels = labels.set_index(barcode_col).drop(columns=sample_col).add_prefix("manual_")
    joined = obs.join(labels, how="left", validate="one_to_one")
    print("Manual label barcode matches:", obs.index.isin(labels.index).sum(), "/", len(obs))
    return joined


def run_manifest(model_dir, settings):
    model_dir = Path(model_dir)
    digest = hashlib.sha256()
    with (model_dir / "best_model.pt").open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(chunk)
    versions = {}
    for package in ["scgpt", "torch", "numpy", "pandas", "scipy", "anndata"]:
        versions[package] = importlib.metadata.version(package)
    return {"settings": settings, "versions": versions,
            "upstream_commit": UPSTREAM_COMMIT, "checkpoint_sha256": digest.hexdigest(),
            "config": json.loads((model_dir / "args.json").read_text()),
            "limitations": ["one sample; CopyKAT labels are provisional",
                "expressed-gene-only targets; observed-only quantile boundaries",
                "reference baseline uses complete reference-cell quantiles",
                "marker panels are descriptive, not cell-type or malignancy labels",
                "CPU translation numerically checked against upstream source with a CPU FlashAttention substitute",
                "not validated on the user's actual checkpoint until strict loading and Nibi execution succeed"]}
