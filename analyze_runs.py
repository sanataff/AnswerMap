"""
Paper numbers from SAVED runs. No GPU.

  python analyze_runs.py --agreement runs/agreement.json
  python analyze_runs.py --deletion  runs/deletion.json

Agreement: per-method NSS / AUC / distance / Pearson r from the saved summary
(the same numbers exp_agreement prints at the end of a run).

Deletion: answer-change rates and gaps vs the random control, the map-maximum
grading (probe-region flip rate by peak tercile, plus the peak -> flip
ROC-AUC), and the image-reliance split (probe flips on image-used vs
image-unused rows).
"""
import argparse, json
import numpy as np


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos = int(y.sum()); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def show_agreement(path):
    d = json.load(open(path))
    summ = d.get("summary", {})
    n = len([r for r in d.get("results", []) if "gen" in r])
    print(f"=== agreement ({path}, n={n}) ===")
    print(f"  {'method':<14}{'NSS':>7}{'AUC':>8}{'dist':>8}{'r':>8}")
    for m, s in summ.items():
        if m == "center" or not isinstance(s, dict) or "mean_dist" not in s:
            continue
        print(f"  {m:<14}{s.get('nss', float('nan')):>7.2f}"
              f"{s.get('auc', float('nan')):>8.3f}"
              f"{s['mean_dist']:>8.3f}{s.get('r', float('nan')):>8.3f}")
    if "center" in summ:
        print(f"  {'centre prior':<14}{'--':>7}{'--':>8}"
              f"{summ['center']['mean_dist']:>8.3f}{'--':>8}")


def show_deletion(path):
    d = json.load(open(path))
    rows = [r for r in d["results"] if r.get("correct_full") == 1]
    fkeys = sorted({k for r in rows for k in r if k.startswith("flip_")})
    print(f"=== deletion ({path}, n correct with full image = {len(rows)}) ===")
    rand = np.asarray([r.get("flip_random", 0) for r in rows], float)
    dr = 100 * float(rand.mean())
    print(f"  {'region':<26}{'answer change %':>17}{'vs random':>11}")
    for k in fkeys:
        f = np.asarray([r[k] for r in rows if k in r], float)
        print(f"  {k[5:]:<26}{100*f.mean():>16.1f}%{100*f.mean()-dr:>+10.1f}")

    # the map maximum grades causal strength: flip rate by peak tercile + AUC
    pk = [(r["probe_peak"], r["flip_probe"], r.get("img_needed", -1))
          for r in rows if "probe_peak" in r and "flip_probe" in r]
    if pk:
        peaks = np.asarray([x[0] for x in pk])
        fl = np.asarray([x[1] for x in pk])
        if peaks.std() > 1e-9:
            order = np.argsort(peaks); t = len(pk) // 3
            lo = 100 * fl[order[:t]].mean(); hi = 100 * fl[order[-t:]].mean()
            print(f"\n  map-maximum grading, probe-region flip rate:")
            print(f"    low-peak tercile  {lo:5.1f}%")
            print(f"    high-peak tercile {hi:5.1f}%")
            print(f"    peak -> flip ROC-AUC: {auc(peaks, fl):.3f}  (n={len(pk)})")

        # image-reliance split (whole-image deletion labelled each row)
        rel = [(f, r) for p, f, n_ in pk for r in [n_] if n_ in (0, 1)]
        used = np.asarray([f for f, n_ in rel if n_ == 1], float)
        unused = np.asarray([f for f, n_ in rel if n_ == 0], float)
        if len(used) and len(unused):
            r_un = np.asarray([r.get("flip_random", 0) for r in rows
                               if r.get("img_needed") == 0], float)
            print(f"\n  image-reliance split, probe-region flip rate:")
            print(f"    image USED rows   {100*used.mean():5.1f}%  (n={len(used)})")
            print(f"    image UNUSED rows {100*unused.mean():5.1f}%  (n={len(unused)})"
                  f"   [random there: {100*r_un.mean():.1f}%]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agreement", default=None)
    ap.add_argument("--deletion", default=None)
    a = ap.parse_args()
    if not (a.agreement or a.deletion):
        ap.error("pass --agreement and/or --deletion run json paths")
    if a.agreement:
        show_agreement(a.agreement)
    if a.deletion:
        if a.agreement:
            print()
        show_deletion(a.deletion)


if __name__ == "__main__":
    main()
