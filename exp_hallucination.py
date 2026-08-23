"""
Object hallucination audit (paper 5.1). When the model claims an object is
present, probe where it believes the object is, then apply the deletion test
to that claim.

Two signals per yes-claim, POPE ground truth scores them afterwards:
  1. the deletion signature: a GROUNDED yes dies when its own region is
     deleted and survives a random deletion, a HALLUCINATED yes is fragile
     under any perturbation, nothing in the image was carrying it.
  2. the map maximum: an absent object gives the map nowhere to peak, so a
     LOW peak flags a hallucinated claim (the paper's detector; printed as
     "AUC low-peak -> hallucinated").

The audit needs no labels at run time and runs before generation.

Data: POPE jsonl from prep_vqa --dataset pope,
  {image, question: "Is there a X in the image?", answers: ["yes"|"no"]}

Protocol per row:
  1. ask the POPE question, keep only rows where the model says YES
  2. ground truth labels the claim, gt=yes -> grounded, gt=no -> hallucinated
  3. probe the map for the question, delete the top-mass region, re-ask.
     Also delete a matched random region, re-ask.
  4. report flip-to-no rates for grounded vs hallucinated claims, both
     corruptions, plus the two detector AUCs.

Usage:
  python exp_hallucination.py --data data/pope/data.jsonl \
      --images_dir data/pope/images --cache_dir $CD --limit 0 \
      --out runs/halluc_pope.json
"""
import argparse, json, os, re
import numpy as np


def said_yes(txt):
    t = (txt or "").strip().lower()
    return int(bool(re.match(r"^\W*yes\b", t)))


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos = int(y.sum()); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--region_frac", type=float, default=0.12)
    ap.add_argument("--probe_side", type=int, default=512)
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--out", default="runs/halluc_pope.json")
    a = ap.parse_args()

    from answermap import Config, VLM, probe as probe_op, load_image
    from exp_deletion import corrupt_cells, top_mass_cells, random_cells

    cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                 max_side=a.max_side, min_pixels=a.min_pixels)
    vlm = VLM(cfg)
    cfgp = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                  max_side=a.probe_side, min_pixels=a.min_pixels)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    n_keep = max(1, int(round(a.region_frac * a.K * a.K)))
    print(f"[halluc] auditing YES claims, deleting {n_keep}/{a.K*a.K} cells")

    def yes_on(img, q):
        return said_yes(vlm.ask(f"{q}\nAnswer yes or no.", imgs=[img],
                                max_new_tokens=6))

    out_rows, n_yes = [], 0
    flips_m = {0: [], 1: []}       # keyed by hallucinated (1) vs grounded (0)
    flips_r = {0: [], 1: []}
    peaks = {0: [], 1: []}
    for i, r in enumerate(rows):
        gold = r.get("answers") or r.get("answer")
        gt = (gold[0] if isinstance(gold, list) else gold).strip().lower()
        if gt not in ("yes", "no"):
            continue
        path = os.path.join(a.images_dir, r["image"])
        big = load_image(path, a.max_side); small = load_image(path, a.probe_side)
        q = r["question"]

        if not yes_on(big, q):
            out_rows.append({"image": r["image"], "said_yes": 0}); continue
        n_yes += 1
        hal = int(gt == "no")                       # a YES claim on an absent object

        res = probe_op(vlm, small, q, cfgp)
        M = np.outer(np.asarray(res["c_row"]), np.asarray(res["c_col"]))
        peak = float(M.max())
        pcells = top_mass_cells(M, n_keep)
        rcells = random_cells(a.K, n_keep, pcells, seed=i)

        fm = int(not yes_on(corrupt_cells(big, pcells, a.K, "blank"), q))
        fr = int(not yes_on(corrupt_cells(big, rcells, a.K, "blank"), q))
        flips_m[hal].append(fm); flips_r[hal].append(fr); peaks[hal].append(peak)
        out_rows.append({"image": r["image"], "said_yes": 1, "hallucinated": hal,
                         "flip_map": fm, "flip_random": fr, "peak": round(peak, 4),
                         "category": r.get("category", "")})
        if (i + 1) % 20 == 0 and n_yes:
            g, h = flips_m[0], flips_m[1]
            print(f"[{i+1}/{len(rows)}] yes-claims={n_yes} "
                  f"grounded flip={100*np.mean(g) if g else 0:.0f}% (n={len(g)})  "
                  f"halluc flip={100*np.mean(h) if h else 0:.0f}% (n={len(h)})",
                  flush=True)

    print(f"\n=== Deletion-immunity hallucination audit "
          f"(n={len(out_rows)}, yes-claims={n_yes}) ===")
    print(f"  {'claim type':<24}{'n':>6}{'flip, map region':>18}{'flip, random':>15}")
    print("  " + "-" * 63)
    for hal, name in ((0, "grounded (object real)"), (1, "HALLUCINATED")):
        if flips_m[hal]:
            print(f"  {name:<24}{len(flips_m[hal]):>6}"
                  f"{100*np.mean(flips_m[hal]):>17.1f}%"
                  f"{100*np.mean(flips_r[hal]):>14.1f}%")
    print("\n  Reading: grounded yeses die only when their own region is")
    print("  deleted, hallucinated yeses are fragile under ANY perturbation.")

    # the detector: predict hallucinated iff the claim SURVIVES map deletion
    if flips_m[0] and flips_m[1]:
        surv = [(1 - f) for f in flips_m[0]] + [(1 - f) for f in flips_m[1]]
        lab = [0] * len(flips_m[0]) + [1] * len(flips_m[1])
        tp = sum(s for s, l in zip(surv, lab) if l == 1)
        fp = sum(s for s, l in zip(surv, lab) if l == 0)
        print(f"\n  survives-deletion as the hallucination flag: "
              f"recall {100*tp/max(1,sum(lab)):.1f}%  "
              f"false-positive rate {100*fp/max(1,len(lab)-sum(lab)):.1f}%")
        a_surv = auc(surv, lab)
        a_peak = auc([-p for p in peaks[0]] + [-p for p in peaks[1]], lab)
        print(f"  AUC survives-deletion -> hallucinated : {a_surv:.3f}")
        print(f"  AUC low-peak -> hallucinated (scalar) : {a_peak:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({"n_yes": n_yes, "results": out_rows}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
