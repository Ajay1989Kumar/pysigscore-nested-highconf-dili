#!/usr/bin/env python3
"""
Unsupervised clustering / embedding of pysigscore pathway scores
================================================================
Labels (toxic vs safe) are used **only after** clustering/embedding
for colouring and post-hoc enrichment — never for fitting.

Analyses
  1. PCA (2D) of Hallmark scores per method + stacked multi-method
  2. t-SNE (2D) for non-linear structure
  3. Hierarchical clustering (drugs) + heatmap (GSVA)
  4. k-means (k=2) and agglomerative (k=2) cluster assignments
  5. Separation metrics (post-hoc vs y_hard):
       - ARI, NMI, purity, accuracy (best label permutation)
       - Silhouette using true labels as pseudo-clusters
       - Between/within class distance ratio
       - Fisher exact toxic enrichment of each cluster

Usage (from repo root):
  python3 scripts/unsupervised_clustering.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 13
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RES = ROOT / "results" / "unsupervised"
RES.mkdir(parents=True, exist_ok=True)

# Prefer multi-method matrices from the analysis project if present
EXTERNAL_SCORE_DIR = Path(
    "/Users/ajaykumar/pysigscore-tggates-highconf-dili/results"
)
METHODS = ["GSVA", "ssGSEA", "Mean", "Median", "Z"]

LABELS = DATA / "human_dili_highconf_labels.csv"
OR_CSV = DATA / "rat_or_endpoints.csv"
GSVA_LOCAL = DATA / "scores_log2fc_GSVA_hallmark.csv"

rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid", context="talk")


def load_scores() -> dict[str, pd.DataFrame]:
    scores = {}
    # GSVA always from local bundle
    g = pd.read_csv(GSVA_LOCAL, index_col=0)
    g.index = g.index.astype(str)
    scores["GSVA"] = g

    for m in METHODS:
        if m == "GSVA":
            continue
        p = EXTERNAL_SCORE_DIR / f"scores_log2fc_{m}_hallmark.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0)
            df.index = df.index.astype(str)
            # align to GSVA drugs/columns
            df = df.reindex(index=g.index, columns=g.columns)
            scores[m] = df
    return scores


def load_labels(index: pd.Index) -> pd.Series:
    lab = pd.read_csv(LABELS)
    lab["drug_key"] = lab["drug_key"].astype(str)
    y = lab.set_index("drug_key")["y_hard"].reindex(index).astype(int)
    if y.isna().any():
        raise RuntimeError("Missing labels for some drugs")
    return y


def load_or(index: pd.Index) -> pd.Series:
    if not OR_CSV.exists():
        return pd.Series(0, index=index)
    o = pd.read_csv(OR_CSV)
    o["drug_key"] = o["drug_key"].astype(str)
    o = o.set_index("drug_key")
    cols = [c for c in o.columns if c != "drug_key"]
    bits = o.reindex(index)[cols].fillna(0).astype(int)
    return (bits.max(axis=1) > 0).astype(int)


def zscore_matrix(X: np.ndarray) -> np.ndarray:
    sc = StandardScaler()
    return sc.fit_transform(X)


def best_binary_accuracy(y_true: np.ndarray, clusters: np.ndarray) -> tuple[float, np.ndarray]:
    """Map 2 clusters → labels by max accuracy (post-hoc)."""
    y_true = y_true.astype(int)
    clusters = clusters.astype(int)
    # try both permutations of cluster→class
    best_acc, best_pred = -1.0, clusters.copy()
    for flip in (False, True):
        pred = clusters.copy()
        if flip:
            pred = 1 - pred
        # map cluster ids {0,1} already as class if same coding
        acc = float((pred == y_true).mean())
        # also try remapping by majority vote
        pred2 = np.zeros_like(clusters)
        for c in np.unique(clusters):
            mask = clusters == c
            maj = int(np.round(y_true[mask].mean())) if mask.sum() else 0
            pred2[mask] = maj
        acc2 = float((pred2 == y_true).mean())
        if acc2 > best_acc:
            best_acc, best_pred = acc2, pred2
        if acc > best_acc:
            best_acc, best_pred = acc, pred
    return best_acc, best_pred


def cluster_purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    y_true = y_true.astype(int)
    clusters = clusters.astype(int)
    n = len(y_true)
    pure = 0
    for c in np.unique(clusters):
        mask = clusters == c
        if not mask.any():
            continue
        # majority label count
        vals, cnts = np.unique(y_true[mask], return_counts=True)
        pure += int(cnts.max())
    return pure / n


def fisher_toxic_enrichment(y: np.ndarray, clusters: np.ndarray) -> list[dict]:
    rows = []
    y = y.astype(int)
    for c in sorted(np.unique(clusters)):
        in_c = clusters == c
        # table: toxic/safe × in/out cluster
        a = int(((y == 1) & in_c).sum())  # toxic in
        b = int(((y == 1) & ~in_c).sum())  # toxic out
        c_ = int(((y == 0) & in_c).sum())  # safe in
        d = int(((y == 0) & ~in_c).sum())  # safe out
        oddsr, p = fisher_exact([[a, b], [c_, d]], alternative="two-sided")
        rows.append(
            {
                "cluster": int(c),
                "n": int(in_c.sum()),
                "n_toxic": a,
                "n_safe": c_,
                "frac_toxic": a / max(in_c.sum(), 1),
                "odds_ratio": float(oddsr),
                "fisher_p": float(p),
            }
        )
    return rows


def separation_metrics(Xz: np.ndarray, y: np.ndarray, tag: str) -> dict:
    """Label-free geometry + post-hoc class separation (labels only for metrics)."""
    y = y.astype(int)
    out = {"feature_set": tag, "n": int(len(y)), "n_toxic": int((y == 1).sum()), "n_safe": int((y == 0).sum())}

    # Silhouette treating true labels as clusters (higher ⇒ better natural separation)
    if len(np.unique(y)) > 1 and len(y) > 2:
        try:
            out["silhouette_true_labels"] = float(silhouette_score(Xz, y, metric="euclidean"))
        except Exception:
            out["silhouette_true_labels"] = np.nan
    else:
        out["silhouette_true_labels"] = np.nan

    # centroid distance ratio
    Xt, Xs = Xz[y == 1], Xz[y == 0]
    ct, cs = Xt.mean(0), Xs.mean(0)
    between = float(np.linalg.norm(ct - cs))
    # mean within-class distance to own centroid
    wt = float(np.mean(np.linalg.norm(Xt - ct, axis=1))) if len(Xt) else np.nan
    ws = float(np.mean(np.linalg.norm(Xs - cs, axis=1))) if len(Xs) else np.nan
    within = np.nanmean([wt, ws])
    out["centroid_distance_between"] = between
    out["mean_within_centroid_dist"] = float(within)
    out["between_over_within"] = float(between / within) if within and within > 0 else np.nan

    # Mann-Whitney on PC1 (after PCA fit on Xz)
    pca1 = PCA(n_components=1, random_state=SEED).fit_transform(Xz).ravel()
    try:
        u, p = mannwhitneyu(pca1[y == 1], pca1[y == 0], alternative="two-sided")
        out["pc1_mannwhitney_U"] = float(u)
        out["pc1_mannwhitney_p"] = float(p)
        out["pc1_auroc_abs"] = float(
            abs(
                # rank AUROC of PC1 vs label (direction-invariant via max(a,1-a))
                __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, pca1)
            )
        )
        a = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, pca1))
        out["pc1_auroc_abs"] = max(a, 1 - a)
    except Exception as e:
        out["pc1_mannwhitney_p"] = np.nan
        out["pc1_auroc_abs"] = np.nan

    # k-means k=2 (unsupervised)
    km = KMeans(n_clusters=2, random_state=SEED, n_init=20)
    c_km = km.fit_predict(Xz)
    out["kmeans_silhouette"] = float(silhouette_score(Xz, c_km))
    out["kmeans_ARI"] = float(adjusted_rand_score(y, c_km))
    out["kmeans_NMI"] = float(normalized_mutual_info_score(y, c_km))
    out["kmeans_purity"] = float(cluster_purity(y, c_km))
    acc, _ = best_binary_accuracy(y, c_km)
    out["kmeans_posthoc_accuracy"] = acc

    # agglomerative k=2 (euclidean, ward)
    ag = AgglomerativeClustering(n_clusters=2, linkage="ward")
    c_ag = ag.fit_predict(Xz)
    out["agglo_silhouette"] = float(silhouette_score(Xz, c_ag))
    out["agglo_ARI"] = float(adjusted_rand_score(y, c_ag))
    out["agglo_NMI"] = float(normalized_mutual_info_score(y, c_ag))
    out["agglo_purity"] = float(cluster_purity(y, c_ag))
    acc, _ = best_binary_accuracy(y, c_ag)
    out["agglo_posthoc_accuracy"] = acc

    return out, c_km, c_ag, pca1


def plot_embedding(
    xy: np.ndarray,
    y: np.ndarray,
    or_vec: np.ndarray,
    title: str,
    path: Path,
    xlab: str,
    ylab: str,
    cluster: np.ndarray | None = None,
):
    fig, axes = plt.subplots(1, 2 if cluster is not None else 1, figsize=(14 if cluster is not None else 7, 6))
    if cluster is None:
        axes = [axes]

    colors = np.where(y == 1, "#d62728", "#1f77b4")
    markers = np.where(or_vec == 1, "o", "s")

    ax = axes[0]
    for m_flag, marker in ((0, "s"), (1, "o")):
        mask = or_vec == m_flag
        ax.scatter(
            xy[mask & (y == 1), 0],
            xy[mask & (y == 1), 1],
            c="#d62728",
            marker=marker,
            s=70,
            edgecolors="k",
            linewidths=0.4,
            alpha=0.85,
            label="Toxic" if m_flag == 1 else None,
        )
        ax.scatter(
            xy[mask & (y == 0), 0],
            xy[mask & (y == 0), 1],
            c="#1f77b4",
            marker=marker,
            s=110,
            edgecolors="k",
            linewidths=0.6,
            alpha=0.95,
            label="Safe" if m_flag == 1 else None,
        )
    # legend manually
    handles = [
        Patch(facecolor="#d62728", edgecolor="k", label="Toxic (y=1)"),
        Patch(facecolor="#1f77b4", edgecolor="k", label="Safe (y=0)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=10, label="OR+"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=10, label="OR−"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=10, frameon=True)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title + "\n(coloured by label post-hoc; not used in fit)")

    if cluster is not None:
        ax2 = axes[1]
        cmap = np.array(["#2ca02c", "#ff7f0e"])
        ax2.scatter(xy[:, 0], xy[:, 1], c=cmap[cluster.astype(int)], s=70, edgecolors="k", linewidths=0.4, alpha=0.85)
        # overlay safe as stars
        ax2.scatter(
            xy[y == 0, 0],
            xy[y == 0, 1],
            facecolors="none",
            edgecolors="black",
            s=220,
            linewidths=1.5,
            marker="o",
            label="Safe drugs",
        )
        ax2.set_xlabel(xlab)
        ax2.set_ylabel(ylab)
        ax2.set_title("Same embedding\ncolour = unsupervised k-means (k=2)")
        ax2.legend(loc="best", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_clustered(df: pd.DataFrame, y: pd.Series, path: Path, title: str):
    """Hierarchical clustering of drugs (rows) and pathways (cols)."""
    X = zscore_matrix(df.values.astype(float))
    # distance between drugs
    Z = linkage(X, method="ward")
    # order
    dendro = dendrogram(Z, no_plot=True)
    row_order = dendro["leaves"]
    # cluster pathways
    Zc = linkage(X.T, method="ward")
    col_leaves = dendrogram(Zc, no_plot=True)["leaves"]

    mat = X[row_order][:, col_leaves]
    drugs = df.index.to_numpy()[row_order]
    paths = df.columns.to_numpy()[col_leaves]
    y_ord = y.loc[drugs].values

    # short pathway names
    path_labels = [p.replace("HALLMARK_", "")[:28] for p in paths]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.03, 1.0], wspace=0.05)
    ax_lab = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    # label strip: toxic red, safe blue
    lab_col = np.where(y_ord == 1, 1.0, 0.0).reshape(-1, 1)
    ax_lab.imshow(lab_col, aspect="auto", cmap=plt.cm.colors.ListedColormap(["#1f77b4", "#d62728"]), vmin=0, vmax=1)
    ax_lab.set_xticks([])
    ax_lab.set_yticks([])
    ax_lab.set_ylabel("Drugs (ward hierarchical order)", fontsize=11)
    ax_lab.set_title("T/S", fontsize=9)

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_yticks(range(len(drugs)))
    ax.set_yticklabels(
        [f"{'●' if yy == 1 else '○'} {d}" for d, yy in zip(drugs, y_ord)],
        fontsize=6,
        fontfamily="monospace",
    )
    ax.set_xticks(range(len(path_labels)))
    ax.set_xticklabels(path_labels, rotation=90, fontsize=7)
    ax.set_title(title + "\n● toxic  ○ safe  (labels not used for clustering)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Row z-score of pathway score")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # cut tree into 2 clusters for metrics
    cl = fcluster(Z, t=2, criterion="maxclust")
    # map leaf order clusters back to original drug order
    clusters_orig = np.zeros(len(df), dtype=int)
    for new_i, orig_i in enumerate(row_order):
        clusters_orig[orig_i] = cl[new_i] - 1  # 0/1
    return clusters_orig


def plot_dendrogram(df: pd.DataFrame, y: pd.Series, path: Path, title: str):
    X = zscore_matrix(df.values.astype(float))
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    # color labels
    drugs = list(df.index)
    dn = dendrogram(Z, labels=drugs, leaf_rotation=90, leaf_font_size=7, ax=ax, color_threshold=0)
    # recolor leaf labels
    ylbs = ax.get_xmajorticklabels()
    for lbl in ylbs:
        name = lbl.get_text()
        if name in y.index:
            lbl.set_color("#d62728" if int(y.loc[name]) == 1 else "#1f77b4")
            lbl.set_fontweight("bold" if int(y.loc[name]) == 0 else "normal")
    ax.set_title(title + "\nleaf colour: red=toxic, blue=safe (post-hoc)")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metrics_df: pd.DataFrame, path: Path):
    plot_df = metrics_df.copy()
    cols = [
        "silhouette_true_labels",
        "between_over_within",
        "pc1_auroc_abs",
        "kmeans_ARI",
        "kmeans_purity",
        "agglo_ARI",
        "agglo_purity",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    pairs = [
        (axes[0, 0], "silhouette_true_labels", "Silhouette (true toxic/safe as groups)"),
        (axes[0, 1], "pc1_auroc_abs", "PC1 separation (AUROC abs)"),
        (axes[1, 0], "kmeans_ARI", "k-means k=2 vs labels (ARI)"),
        (axes[1, 1], "kmeans_purity", "k-means k=2 purity"),
    ]
    for ax, col, title in pairs:
        sns.barplot(data=plot_df, x="feature_set", y=col, ax=ax, palette="viridis")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle("Unsupervised separation of toxic vs safe\n(using pysigscore Hallmark scores)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("Unsupervised clustering of pysigscore scores")
    print("=" * 70)

    scores = load_scores()
    print(f"Methods loaded: {list(scores.keys())}")
    ref = scores["GSVA"]
    y = load_labels(ref.index)
    or_vec = load_or(ref.index).values
    drugs = ref.index.to_list()

    # save label strip
    pd.DataFrame({"drug_key": drugs, "y_hard": y.values, "OR": or_vec}).to_csv(
        RES / "cohort.csv", index=False
    )

    all_metrics = []
    cluster_tables = []
    fisher_rows = []

    # Per-method + stacked
    feature_sets: dict[str, pd.DataFrame] = {m: scores[m] for m in scores}
    if len(scores) > 1:
        # stack columns with method prefix
        parts = []
        for m, df in scores.items():
            d = df.copy()
            d.columns = [f"{m}::{c}" for c in d.columns]
            parts.append(d)
        feature_sets["STACKED"] = pd.concat(parts, axis=1)

    for tag, df in feature_sets.items():
        print(f"\n--- {tag}  shape={df.shape} ---", flush=True)
        X = df.values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        Xz = zscore_matrix(X)

        met, c_km, c_ag, pca1 = separation_metrics(Xz, y.values, tag)
        all_metrics.append(met)
        print(
            f"  silhouette(labels)={met['silhouette_true_labels']:.3f}  "
            f"PC1|AUROC|={met['pc1_auroc_abs']:.3f}  "
            f"kmeans ARI={met['kmeans_ARI']:.3f} purity={met['kmeans_purity']:.3f}  "
            f"between/within={met['between_over_within']:.3f}",
            flush=True,
        )

        for cname, cl in (("kmeans", c_km), ("agglo", c_ag)):
            for row in fisher_toxic_enrichment(y.values, cl):
                fisher_rows.append({"feature_set": tag, "algorithm": cname, **row})
            cluster_tables.append(
                pd.DataFrame(
                    {
                        "drug_key": drugs,
                        "y_hard": y.values,
                        "OR": or_vec,
                        "feature_set": tag,
                        "algorithm": cname,
                        "cluster": cl,
                    }
                )
            )

        # PCA embedding
        pca = PCA(n_components=2, random_state=SEED)
        xy = pca.fit_transform(Xz)
        ev = pca.explained_variance_ratio_
        plot_embedding(
            xy,
            y.values,
            or_vec,
            title=f"PCA — pysigscore {tag} Hallmark scores\nPC1 {ev[0]*100:.1f}% · PC2 {ev[1]*100:.1f}%",
            path=RES / f"pca_{tag}.png",
            xlab=f"PC1 ({ev[0]*100:.1f}%)",
            ylab=f"PC2 ({ev[1]*100:.1f}%)",
            cluster=c_km,
        )
        # save coords
        pd.DataFrame(
            {
                "drug_key": drugs,
                "y_hard": y.values,
                "OR": or_vec,
                "PC1": xy[:, 0],
                "PC2": xy[:, 1],
                "kmeans": c_km,
                "agglo": c_ag,
            }
        ).to_csv(RES / f"embedding_pca_{tag}.csv", index=False)

        # t-SNE
        perplexity = min(30, max(5, len(drugs) // 4))
        tsne = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        xy_t = tsne.fit_transform(Xz)
        plot_embedding(
            xy_t,
            y.values,
            or_vec,
            title=f"t-SNE — pysigscore {tag} Hallmark scores",
            path=RES / f"tsne_{tag}.png",
            xlab="t-SNE 1",
            ylab="t-SNE 2",
            cluster=c_km,
        )
        pd.DataFrame(
            {
                "drug_key": drugs,
                "y_hard": y.values,
                "OR": or_vec,
                "TSNE1": xy_t[:, 0],
                "TSNE2": xy_t[:, 1],
                "kmeans": c_km,
            }
        ).to_csv(RES / f"embedding_tsne_{tag}.csv", index=False)

    # Detailed GSVA hierarchical heatmap + dendrogram
    print("\n--- hierarchical GSVA ---", flush=True)
    cl_h = plot_heatmap_clustered(
        feature_sets["GSVA"],
        y,
        RES / "heatmap_GSVA_clustered.png",
        "Hierarchical clustering of drugs × Hallmark GSVA (pysigscore)",
    )
    plot_dendrogram(
        feature_sets["GSVA"],
        y,
        RES / "dendrogram_GSVA.png",
        "Ward dendrogram — pysigscore GSVA Hallmark scores",
    )
    met_h = {
        "feature_set": "GSVA_hierarchical",
        "agglo_ARI": float(adjusted_rand_score(y.values, cl_h)),
        "agglo_NMI": float(normalized_mutual_info_score(y.values, cl_h)),
        "agglo_purity": float(cluster_purity(y.values, cl_h)),
    }
    acc, _ = best_binary_accuracy(y.values, cl_h)
    met_h["agglo_posthoc_accuracy"] = acc
    print(f"  hierarchical ARI={met_h['agglo_ARI']:.3f} purity={met_h['agglo_purity']:.3f}", flush=True)
    for row in fisher_toxic_enrichment(y.values, cl_h):
        fisher_rows.append({"feature_set": "GSVA", "algorithm": "hierarchical_ward", **row})

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RES / "separation_metrics.csv", index=False)
    pd.DataFrame(fisher_rows).to_csv(RES / "cluster_toxic_enrichment.csv", index=False)
    pd.concat(cluster_tables, ignore_index=True).to_csv(RES / "cluster_assignments.csv", index=False)
    plot_metrics_bar(metrics_df, RES / "metrics_comparison.png")

    # Safe-drug focus table: where do safes sit?
    safe_keys = y[y == 0].index.tolist()
    focus = []
    pca = PCA(n_components=2, random_state=SEED)
    Xz = zscore_matrix(feature_sets["GSVA"].values.astype(float))
    xy = pca.fit_transform(Xz)
    km = KMeans(n_clusters=2, random_state=SEED, n_init=20).fit_predict(Xz)
    for i, d in enumerate(drugs):
        if d in safe_keys:
            focus.append(
                {
                    "drug_key": d,
                    "y_hard": 0,
                    "OR": int(or_vec[i]),
                    "PC1": xy[i, 0],
                    "PC2": xy[i, 1],
                    "kmeans_cluster": int(km[i]),
                    "pc1_percentile_among_all": float((xy[:, 0] <= xy[i, 0]).mean() * 100),
                }
            )
    pd.DataFrame(focus).to_csv(RES / "safe_drugs_in_GSVA_space.csv", index=False)

    # Human-readable report
    lines = []
    lines.append("Unsupervised clustering of pysigscore Hallmark scores")
    lines.append("=" * 70)
    lines.append("Labels (toxic/safe) NOT used for PCA, t-SNE, k-means, hierarchical clustering.")
    lines.append(f"Cohort: {len(drugs)} drugs ({int((y==1).sum())} toxic / {int((y==0).sum())} safe)")
    lines.append(f"Feature sets: {list(feature_sets.keys())}")
    lines.append("")
    lines.append("How to read metrics")
    lines.append("-" * 70)
    lines.append("  silhouette_true_labels : geometry of toxic vs safe as two groups (−1..1; >0 = separated)")
    lines.append("  pc1_auroc_abs          : how well PC1 ranks toxic vs safe (0.5=chance, 1=perfect)")
    lines.append("  between_over_within    : centroid distance / mean within-class radius")
    lines.append("  kmeans_ARI             : agreement of unsupervised k=2 with true labels (0=chance)")
    lines.append("  kmeans_purity          : majority-label purity of unsupervised clusters")
    lines.append("")
    lines.append("Results by feature set")
    lines.append("-" * 70)
    show = [
        "feature_set",
        "silhouette_true_labels",
        "pc1_auroc_abs",
        "between_over_within",
        "kmeans_ARI",
        "kmeans_NMI",
        "kmeans_purity",
        "kmeans_posthoc_accuracy",
        "agglo_ARI",
        "agglo_purity",
        "pc1_mannwhitney_p",
    ]
    for _, r in metrics_df.sort_values("silhouette_true_labels", ascending=False).iterrows():
        lines.append(
            f"  {r['feature_set']:10s}  sil={r['silhouette_true_labels']:+.3f}  "
            f"|AUROC_PC1|={r['pc1_auroc_abs']:.3f}  B/W={r['between_over_within']:.2f}  "
            f"kmeans ARI={r['kmeans_ARI']:.3f} purity={r['kmeans_purity']:.3f} acc={r['kmeans_posthoc_accuracy']:.3f}  "
            f"MWU_p={r['pc1_mannwhitney_p']:.2e}"
        )
    lines.append("")
    lines.append(f"GSVA hierarchical (ward, k=2): ARI={met_h['agglo_ARI']:.3f}  "
                 f"purity={met_h['agglo_purity']:.3f}  posthoc_acc={met_h['agglo_posthoc_accuracy']:.3f}")
    lines.append("")
    # Best method summary
    best = metrics_df.sort_values("silhouette_true_labels", ascending=False).iloc[0]
    lines.append("Interpretation")
    lines.append("-" * 70)
    sil = best["silhouette_true_labels"]
    ari = best["kmeans_ARI"]
    if sil > 0.15 and ari > 0.1:
        verdict = "MODERATE unsupervised separation of toxic vs safe in pathway space."
    elif sil > 0.05 or ari > 0.05:
        verdict = "WEAK–MODERATE separation: structure exists but toxic/safe are not clean clusters."
    else:
        verdict = (
            "LITTLE unsupervised separation: toxic and safe largely intermixed in pathway space; "
            "supervised top-k/OR fusion is what drives predictive AUROC."
        )
    lines.append(f"  Best feature set by silhouette: {best['feature_set']}")
    lines.append(f"  Verdict: {verdict}")
    lines.append("")
    lines.append("  Note: with only 7 safe drugs, ARI/purity are noisy; inspect PCA/t-SNE figures")
    lines.append("  and safe_drugs_in_GSVA_space.csv for where safes land among toxics.")
    lines.append("")
    lines.append(f"Figures and tables → {RES}")
    report = "\n".join(lines) + "\n"
    (RES / "unsupervised_report.txt").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
