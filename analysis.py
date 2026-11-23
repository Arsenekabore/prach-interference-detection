"""analysis.py -- statistics, LaTeX tables and figures from the CSV files written by experiments.py."""
import os, json, itertools
import numpy as np
import pandas as pd
from scipy import stats as st

ORDER = ["Correlation", "XGBoost", "CNN", "Conv1D", "LSTM", "Transformer", "Conv1D-IQ"]
# validated categorical palette (fixed order); the classical baseline is a neutral gray
COLORS = {"XGBoost": "#2a78d6", "CNN": "#eb6834", "Conv1D": "#1baf7a", "LSTM": "#eda100",
          "Transformer": "#e87ba4", "Conv1D-IQ": "#008300", "Correlation": "#6f6e69"}
MARKERS = {"XGBoost": "o", "CNN": "s", "Conv1D": "^", "LSTM": "D", "Transformer": "v", "Conv1D-IQ": "P", "Correlation": "x"}


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def holm(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load_runs(out="results"):
    """Read runs.csv robustly: rows written without the baseline-only 'thresholds' column
    (all learning models) have one field fewer than the header, so the field is re-inserted."""
    import csv
    with open(os.path.join(out, "runs.csv"), newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    k = head.index("thresholds") if "thresholds" in head else None
    fixed = []
    for r in body:
        if k is not None and len(r) == len(head) - 1:
            r = r[:k] + [""] + r[k:]
        if len(r) != len(head):
            raise ValueError(f"unexpected row length {len(r)}: {r[:3]}")
        fixed.append(r)
    d = pd.DataFrame(fixed, columns=head)
    for c in d.columns:
        if c not in ("model", "thresholds"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def summarise(out="results"):
    d = load_runs(out)
    models = [m for m in ORDER if m in d["model"].unique()]
    cols = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
            "f1_none", "f1_intra", "f1_inter", "bin_precision", "bin_recall", "bin_f1",
            "test_unseen_roots_accuracy", "test_unseen_roots_macro_f1", "test_EPA5_accuracy", "test_EPA5_macro_f1",
            "fit_s", "infer_ms_per_sample", "preproc_ms_per_sample", "n_params", "epochs"]
    rows = []
    for m in models:
        g = d[d.model == m]
        r = dict(model=m, n_runs=len(g))
        for c in cols:
            r[c + "_mean"], r[c + "_std"], r[c + "_ci95"] = g[c].mean(), g[c].std(ddof=1), ci95(g[c])
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, "summary.csv"), index=False)

    # ---- significance: Friedman over runs + pairwise Wilcoxon signed-rank (Holm-corrected)
    res = {}
    for metric in ["macro_f1", "accuracy"]:
        W = d.pivot(index="run", columns="model", values=metric)[models].dropna()
        out_m = {"n_runs": int(len(W))}
        if len(W) >= 3 and len(models) >= 3:
            fr = st.friedmanchisquare(*[W[m] for m in models])
            out_m["friedman_chi2"], out_m["friedman_p"] = float(fr.statistic), float(fr.pvalue)
            out_m["mean_rank"] = {m: float(v) for m, v in W.rank(axis=1, ascending=False).mean().items()}
        pairs, pv, diffs = [], [], []
        for a, b in itertools.combinations(models, 2):
            x, y = W[a].values, W[b].values
            if np.allclose(x, y):
                p = 1.0
            else:
                p = float(st.wilcoxon(x, y).pvalue) if len(W) >= 5 else np.nan
            pairs.append((a, b)); pv.append(p); diffs.append(float(np.mean(x - y)))
        adj = holm(np.nan_to_num(pv, nan=1.0))
        out_m["pairwise"] = [dict(a=a, b=b, mean_diff=df_, p=p, p_holm=float(q))
                             for (a, b), df_, p, q in zip(pairs, diffs, pv, adj)]
        res[metric] = out_m
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    pd.DataFrame(res["macro_f1"]["pairwise"]).to_csv(os.path.join(out, "pairwise_macro_f1.csv"), index=False)
    return S, res


def _pm(S, m, c, scale=100, nd=2):
    r = S[S.model == m].iloc[0]
    return f"{scale*r[c+'_mean']:.{nd}f} $\\pm$ {scale*r[c+'_std']:.{nd}f}"


def latex_tables(out="results"):
    S = pd.read_csv(os.path.join(out, "summary.csv"))
    models = [m for m in ORDER if m in S.model.values]
    L = []
    L.append("% Table: detection performance (mean +- std over runs, %)")
    L.append("\\begin{tabular}{lcccccc}\n\\toprule")
    L.append("Model & Accuracy & Macro-F1 & Weighted-F1 & F1 (none) & F1 (intra) & F1 (inter) \\\\\n\\midrule")
    for m in models:
        L.append(f"{m} & " + " & ".join(_pm(S, m, c) for c in
                 ["accuracy", "macro_f1", "weighted_f1", "f1_none", "f1_intra", "f1_inter"]) + " \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n")
    L.append("% Table: robustness (macro-F1 %, mean +- std)")
    L.append("\\begin{tabular}{lccc}\n\\toprule\nModel & Nominal test & Unseen inter-cell roots & EPA-5\\,Hz channel \\\\\n\\midrule")
    for m in models:
        L.append(f"{m} & " + " & ".join(_pm(S, m, c) for c in
                 ["macro_f1", "test_unseen_roots_macro_f1", "test_EPA5_macro_f1"]) + " \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n")
    L.append("% Table: computational cost (mean +- std)")
    L.append("\\begin{tabular}{lcccc}\n\\toprule\nModel & Trainable params & Training time [s] & Inference [ms/sample] & Pre-processing [ms/sample] \\\\\n\\midrule")
    for m in models:
        r = S[S.model == m].iloc[0]
        L.append(f"{m} & {r['n_params_mean']:,.0f} & {r['fit_s_mean']:.1f} $\\pm$ {r['fit_s_std']:.1f} & "
                 f"{r['infer_ms_per_sample_mean']:.3f} $\\pm$ {r['infer_ms_per_sample_std']:.3f} & "
                 f"{r['preproc_ms_per_sample_mean']:.3f} \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, "tables.tex"), "w").write("\n".join(L))


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3dd", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8980", "axes.labelcolor": "#2b2b29",
                         "xtick.color": "#55544f", "ytick.color": "#55544f", "legend.frameon": False,
                         "lines.linewidth": 2.0, "lines.markersize": 5, "pdf.fonttype": 42,
                         "savefig.bbox": "tight"})


def _curve(pc, by, models, ax, metric="accuracy"):
    for m in models:
        g = pc[(pc.model == m)]
        per_run = g.groupby(["run", by])[["n", "n_correct"]].sum().reset_index()
        per_run["acc"] = per_run.n_correct / per_run.n
        agg = per_run.groupby(by)["acc"].agg(["mean", lambda x: ci95(x)])
        agg.columns = ["mean", "ci"]
        x = agg.index.values
        ax.plot(x, 100 * agg["mean"], marker=MARKERS[m], color=COLORS[m], label=m,
                linestyle="--" if m == "Correlation" else "-")
        lo = np.clip(100 * (agg["mean"] - agg["ci"]), 0, 100)
        hi = np.clip(100 * (agg["mean"] + agg["ci"]), 0, 100)
        ax.fill_between(x, lo, hi, color=COLORS[m], alpha=0.12, linewidth=0)
    ax.set_ylim(0, 100)


def figures(out="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    fd = os.path.join(out, "figures")
    os.makedirs(fd, exist_ok=True)
    d = load_runs(out)
    pc = pd.read_csv(os.path.join(out, "per_condition.csv"))
    models = [m for m in ORDER if m in d.model.unique()]
    pct = pc[pc.test_set == "test"]

    # 1. accuracy vs SNR
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    _curve(pct, "snr_db", models, ax)
    ax.set_xlabel("SNR [dB]"); ax.set_ylabel("Accuracy [%] (mean, 95% CI)")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(fd, "acc_vs_snr.pdf")); plt.close(fig)

    # 2. accuracy vs ISR (interference classes only, all SNRs)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    _curve(pct, "isr_db", models, ax)
    ax.set_xlabel("ISR bin [dB] (all classes)"); ax.set_ylabel("Accuracy [%] (mean, 95% CI)")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(fd, "acc_vs_isr.pdf")); plt.close(fig)

    # 3. per-class recall vs SNR (one panel per class)
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    for k, c in enumerate(["none", "intra", "inter"]):
        for m in models:
            g = pct[pct.model == m].groupby(["run", "snr_db"])[[f"n_{c}", f"correct_{c}"]].sum().reset_index()
            g["r"] = g[f"correct_{c}"] / g[f"n_{c}"]
            a = g.groupby("snr_db")["r"].mean()
            axs[k].plot(a.index, 100 * a.values, marker=MARKERS[m], color=COLORS[m], label=m,
                        linestyle="--" if m == "Correlation" else "-")
        axs[k].set_title({"none": "No interference", "intra": "Intra-cell", "inter": "Inter-cell"}[c])
        axs[k].set_xlabel("SNR [dB]")
    axs[0].set_ylabel("Recall [%]")
    axs[-1].legend(fontsize=7, ncol=1)
    fig.savefig(os.path.join(fd, "recall_per_class_vs_snr.pdf")); plt.close(fig)

    # 4. confusion matrices (row-normalised, summed over runs)
    cm = pd.read_csv(os.path.join(out, "confusion.csv"))
    labs = ["none", "intra", "inter"]
    n = len(models)
    fig, axs = plt.subplots(1, n, figsize=(2.0 * n, 2.3))
    for ax, m in zip(np.atleast_1d(axs), models):
        M = cm[cm.model == m].pivot_table(index="true", columns="pred", values="count", aggfunc="sum").reindex(index=labs, columns=labs).values
        Mn = M / M.sum(axis=1, keepdims=True)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100*Mn[i,j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.6 else "#2b2b29")
        ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(labs if ax is np.atleast_1d(axs)[0] else [], fontsize=7)
        ax.set_title(m, fontsize=8); ax.grid(False)
    np.atleast_1d(axs)[0].set_ylabel("True class")
    fig.supxlabel("Predicted class", fontsize=8)
    fig.savefig(os.path.join(fd, "confusion_matrices.pdf")); plt.close(fig)

    # 5. macro-F1 with 95% CI
    S = pd.read_csv(os.path.join(out, "summary.csv")).set_index("model").loc[models]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(models))
    ax.bar(x, 100 * S["macro_f1_mean"], yerr=100 * S["macro_f1_ci95"], color=[COLORS[m] for m in models],
           width=0.6, capsize=3, edgecolor="white", linewidth=2)
    for xi, v in zip(x, S["macro_f1_mean"]):
        ax.text(xi, 100 * v + 1, f"{100*v:.1f}", ha="center", fontsize=7, color="#2b2b29")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, fontsize=8)
    ax.set_ylabel("Macro-F1 [%] (mean, 95% CI)")
    ax.set_ylim(max(0, 100 * S["macro_f1_mean"].min() - 15), 100)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(fd, "macro_f1_ci.pdf")); plt.close(fig)

    # 6. computational cost (two separate charts, log scale)
    for col, lab, fn in [("fit_s", "Training time [s]", "fit_time.pdf"),
                         ("infer_ms_per_sample", "Inference latency [ms/sample]", "inference_time.pdf")]:
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        ax.bar(x, S[col + "_mean"], yerr=S[col + "_std"], color=[COLORS[m] for m in models], width=0.6,
               capsize=3, edgecolor="white", linewidth=2)
        ax.set_yscale("log"); ax.set_ylabel(lab)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, fontsize=8)
        ax.grid(axis="x", visible=False)
        fig.savefig(os.path.join(fd, fn)); plt.close(fig)

    # 7. learning curves of run 1
    hs = [m for m in models if os.path.exists(os.path.join(out, f"history_{m}.csv"))]
    if hs:
        fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0))
        for m in hs:
            h = pd.read_csv(os.path.join(out, f"history_{m}.csv"))
            axs[0].plot(h.epoch, h.loss, color=COLORS[m], linestyle=":", linewidth=1.5)
            axs[0].plot(h.epoch, h.val_loss, color=COLORS[m], label=m)
            axs[1].plot(h.epoch, 100 * h.accuracy, color=COLORS[m], linestyle=":", linewidth=1.5)
            axs[1].plot(h.epoch, 100 * h.val_accuracy, color=COLORS[m], label=m)
        axs[0].set_ylabel("Loss"); axs[1].set_ylabel("Accuracy [%]")
        for a in axs:
            a.set_xlabel("Epoch")
        axs[1].legend(fontsize=7, title="solid: validation, dotted: training", title_fontsize=7)
        fig.savefig(os.path.join(fd, "learning_curves.pdf")); plt.close(fig)


def full_report(out="results"):
    S, res = summarise(out)
    latex_tables(out)
    figures(out)
    return S, res
