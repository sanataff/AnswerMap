"""
Paper figures from SAVED runs. No GPU.

  python make_figures.py layers   --in runs/attn_layer_sweep.json --out figs/
  python make_figures.py scatter  --in runs/agreement.json        --out figs/
  python make_figures.py budget                                   --out figs/

layers  -> figs/attn_layer_sweep.pdf  (agreement of each decoder layer)
scatter -> figs/scatter_agreement.pdf (map expectation vs generated point)
budget  -> figs/budget_curves.pdf     (query budget, refine vs compose; reads
           the run files listed in BUDGET_MANIFEST, missing files are skipped)
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_layers(path, out):
    d = json.load(open(path))
    L = [s["layer"] for s in d["layers"]]
    r = [s["dev_corr"] for s in d["layers"]]
    best = d["best_layer"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(0, color="0.75", lw=0.8)
    ax.plot(L, r, marker="o", ms=3, lw=1.2, color="#444444")
    ax.scatter([best], [r[best]], s=60, zorder=5, color="#c0392b",
               label=f"best layer {best} (r={r[best]:.2f})")
    ax.scatter([L[-1]], [r[-1]], s=60, zorder=5, marker="s", color="#2c5f8a",
               label=f"last layer (r={r[-1]:.2f})")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("Pearson r with generated point")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Attention agreement by layer, mid-stack band, starved tail",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"attn_layer_sweep.{ext}"), dpi=200)
    print(f"[figs] attn_layer_sweep.pdf/png -> {out}")


def fig_scatter(path, out, methods=("probe1", "attn_best")):
    rows = [r for r in json.load(open(path))["results"] if "gen" in r]
    have = [m for m in methods if rows and f"{m}_exp" in rows[0]]
    if not have:
        raise SystemExit("no *_exp fields in this file. Rerun exp_agreement "
                         "(per-row save patch), then re-make this figure.")
    titles = {"probe1": "Probe map (ours)", "attn_best": "Attention (best layer)"}
    fig, axes = plt.subplots(1, len(have), figsize=(3.1 * len(have), 3.2),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, have):
        gx, gy, ex, ey = [], [], [], []
        for r in rows:
            if f"{m}_exp" not in r:
                continue
            W, H = r["wh"]
            gx.append(r["gen"][0] / W); gy.append(r["gen"][1] / H)
            ex.append(r[f"{m}_exp"][0] / W); ey.append(r[f"{m}_exp"][1] / H)
        gx, gy, ex, ey = map(np.asarray, (gx, gy, ex, ey))
        rr = (np.corrcoef(ex, gx)[0, 1] + np.corrcoef(ey, gy)[0, 1]) / 2
        ax.scatter(np.r_[gx, gy], np.r_[ex, ey], s=5, alpha=0.35,
                   color="#2c5f8a", edgecolors="none")
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
        ax.set_title(f"{titles.get(m, m)}   r = {rr:.2f}", fontsize=9)
        ax.set_xlabel("generated point (norm. coord)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    axes[0].set_ylabel("map expectation (norm. coord)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"scatter_agreement.{ext}"), dpi=200)
    print(f"[figs] scatter_agreement.pdf/png -> {out}")


BUDGET_MANIFEST = [
    # (axis, queries, label, file, method_key)  -- missing files are skipped
    ("single", 4,  "K=2",  "runs/agree_MGfix2.json",  "probeMG"),
    ("single", 8,  "K=4",  "runs/agree_K4.json",  "probe1"),
    ("single", 12, "K=6",  "runs/agree_K6.json",  "probe1"),
    ("single", 16, "K=8",  "runs/agree_K8.json",  "probe1"),
    ("single", 20, "K=10", "runs/agree_K10.json", "probe1"),
    ("single", 24, "K=12", "runs/agree_K12.json", "probe1"),
    ("single", 32, "K=16", "runs/agree_K16.json", "probe1"),
    ("single", 34, "K=17", "runs/agree_K17.json", "probe1"),
    ("single", 56, "K=28", "runs/agree_K28.json", "probe1"),
    ("single", 80, "K=40", "runs/agree_K40.json", "probe1"),
    ("multi", 4,  "{2}",           "runs/agree_MGfix2.json",        "probeMG"),
    ("multi", 10, "{2,3}",         "runs/agree_MGfix23.json",       "probeMG"),
    ("multi", 16, "{3,5}",         "runs/agree_MGfix35.json",       "probeMG"),
    ("multi", 20, "{2,3,5}",       "runs/agree_MGfix235.json",      "probeMG"),
    ("multi", 34, "{2,3,5,7}",     "runs/agree_MGfix2357.json",     "probeMG"),
    ("multi", 56, "{2,3,5,7,11}",  "runs/agree_MGfix235711.json",   "probeMG"),
    ("multi", 82, "{2..13}",       "runs/agree_MGfix23571113.json", "probeMG"),
]
EXTRA_POINTS = [
    # plotted as loose markers, not on the curves
    ("nested {2,4,8}", 28, "runs/agree_MG248.json",     "probeMG", "x"),
    ("mean {2,3,5}",   20, "runs/agree_MG235mean.json", "probeMG", "^"),
    ("max {2,3,5}",    20, "runs/agree_MG235max.json",  "probeMG", "v"),
    ("occlusion",      65, "runs/agreement.json",       "occlusion", "s"),
]


def _nss_of(path, key):
    try:
        s = json.load(open(path))["summary"]
        return float(s[key]["nss"])
    except Exception:
        return None


def fig_budget(_, out):
    """Two panels, same y: refine one grid (left) vs compose coarse grids
    (right). The inverted-U on both, composition dominating at every matched
    budget. Missing run files are skipped so the figure grows with the runs."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    titles = {"single": "refine one grid", "multi": "compose coarse grids"}
    for ax, axis in zip(axes, ("single", "multi")):
        pts = [(q, _nss_of(f, k), lab) for (a2, q, lab, f, k) in BUDGET_MANIFEST
               if a2 == axis]
        pts = [(q, v, lab) for (q, v, lab) in pts if v is not None]
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                    lw=1.4, color="#2c5f8a")
            for q, v, lab in pts:
                ax.annotate(lab, (q, v), textcoords="offset points",
                            xytext=(4, 4), fontsize=6.5, color="#444444")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(titles[axis], fontsize=10)
        ax.set_xlabel("queries per map")
    for lab, q, f, k, mk in EXTRA_POINTS:
        v = _nss_of(f, k)
        if v is not None:
            axes[1].scatter([q], [v], marker=mk, s=34, color="#7f8c8d", zorder=5)
            axes[1].annotate(lab, (q, v), textcoords="offset points",
                             xytext=(4, -9), fontsize=6.5, color="#7f8c8d")
    axes[0].set_ylabel("NSS at the model's point")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"budget_curves.{ext}"), dpi=200)
    print(f"[figs] budget_curves.pdf/png -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["layers", "scatter", "budget"])
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.which == "budget":
        fig_budget(a.inp, a.out)
    elif a.which == "layers":
        fig_layers(a.inp, a.out)
    else:
        fig_scatter(a.inp, a.out)


if __name__ == "__main__":
    main()
