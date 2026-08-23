"""
Which layer is attention's BEST? Sweep every decoder layer's attention map and
score its agreement with the model's own generated point -- so the paper's
attention baseline is attention at its best, not a strawman.

WHY. FastV (arXiv 2403.06764) shows deep-layer attention to image tokens is
starved (0.21% of system-prompt attention efficiency; anchor/sink tokens absorb
the rest), and Localization Heads (arXiv 2503.06287) found its useful heads
mid-stack. Reading only the LAST layer invites the review attack "of course that
fails". This sweep is FREE beyond one generation per example: a single forward
with output_attentions returns every layer at once.

For each example: one generation (the model's own point) + one forward for
attentions; then per layer L: mean-head last-token->image map -> centroid
expectation -> agreement metrics across examples (same T1 metrics):

  dev-corr      Pearson corr between (point - center) deviations of map vs
                generation, pooled over x,y -- centrality-invariant
  mean-dist     mean euclidean distance in normalized coords / sqrt(2)
  beats-center  fraction of examples where the map's point is closer to the
                generated point than the image centre is

Report the best layer; use it as raw_attention(layer=BEST) in T1/T2 ("attention
at its best layer"). Appendix figure: dev-corr by layer.

Data: jsonl {image, query, ...} (RefCOCOg from prep_refcoco.py).

Usage:
  python exp_attn_layer_sweep.py --data data/refcocog/refcocog.jsonl \
      --images_dir data/refcocog/images --limit 200 --cache_dir $CD
"""
import argparse, json, os
import numpy as np


def dev_corr(P, G):
    """Pearson corr of deviations-from-center, x and y pooled. P,G: [n,2] in [0,1]."""
    dp = (np.asarray(P) - 0.5).ravel(); dg = (np.asarray(G) - 0.5).ravel()
    if dp.std() < 1e-9 or dg.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(dp, dg)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--side", type=int, default=512,
                    help="image side for the attention forward (matches the "
                         "probe_side used in exp_agreement)")
    ap.add_argument("--gen_side", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="runs/attn_layer_sweep.json")
    a = ap.parse_args()

    import torch
    from answermap import Config, VLM, load_image
    from baselines import generate_point, _grid_and_attn
    cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir,
                 max_side=a.gen_side, min_pixels=a.min_pixels,
                 attn="eager")                    # eager: attentions must exist
    vlm = VLM(cfg)
    cfga = Config(model_name=a.model_name, cache_dir=a.cache_dir,
                  max_side=a.side, min_pixels=a.min_pixels,
                  attn="eager")

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    pts_by_layer = None                  # [L][n,2] map points, normalized
    gens = []                            # [n,2] generated points, normalized
    n_miss = 0
    for i, r in enumerate(rows):
        path = os.path.join(a.images_dir, r["image"])
        big = load_image(path, a.gen_side); small = load_image(path, a.side)
        q = r.get("query") or r.get("question")

        g, _ = generate_point(vlm, big, q, cfg)
        if g is None:                    # parse miss: no reference point, skip
            n_miss += 1; continue
        gn = (g[0] / big.size[0], g[1] / big.size[1])

        attns, pos, Hg, Wg = _grid_and_attn(vlm, small, q, cfga)
        posc = pos.cpu().numpy()
        if pts_by_layer is None:
            pts_by_layer = [[] for _ in range(len(attns))]
        for L in range(len(attns)):
            S = attns[L][0, :, -1, :].float().cpu().numpy().mean(0)[posc]
            M = S.reshape(Hg, Wg)
            B = np.maximum(M - M.mean(), 0.0)    # same centroid as _point_from_map
            if B.sum() <= 0:
                B = M - M.min() + 1e-12
            gy = float((B.sum(1) @ (np.arange(Hg) + .5)) / (B.sum() + 1e-12)) / Hg
            gx = float((B.sum(0) @ (np.arange(Wg) + .5)) / (B.sum() + 1e-12)) / Wg
            pts_by_layer[L].append((gx, gy))
        gens.append(gn)
        del attns
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"[{i+1}/{len(rows)}] kept={len(gens)} parse-miss={n_miss}", flush=True)

    G = np.asarray(gens)
    center_d = float(np.mean(np.linalg.norm(G - 0.5, axis=1)) / np.sqrt(2))
    print(f"\n=== attention layer sweep (n={len(gens)}, miss={n_miss}) ===")
    print(f"  image-centre mean-dist to generation: {center_d:.3f}")
    print(f"\n  {'layer':>5}{'dev-corr':>10}{'mean-dist':>11}{'beats-center':>14}")
    print("  " + "-" * 40)
    stats = []
    for L, pts in enumerate(pts_by_layer):
        P = np.asarray(pts)
        d = np.linalg.norm(P - G, axis=1) / np.sqrt(2)
        dc = np.linalg.norm(G - 0.5, axis=1) / np.sqrt(2)
        row = {"layer": L, "dev_corr": round(dev_corr(P, G), 3),
               "mean_dist": round(float(d.mean()), 3),
               "beats_center": round(float((d < dc).mean()), 3)}
        stats.append(row)
        print(f"  {L:>5}{row['dev_corr']:>10.3f}{row['mean_dist']:>11.3f}"
              f"{row['beats_center']:>14.3f}")
    best = max(stats, key=lambda s: s["dev_corr"])
    print(f"\n  BEST layer by dev-corr: {best['layer']} "
          f"(dev-corr {best['dev_corr']}, mean-dist {best['mean_dist']})")
    print("  -> use raw_attention(layer=BEST) as the 'attention at its best "
          "layer' row in T1/T2. Compare against probe1 dev-corr 0.635 (RefCOCOg).")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({"n": len(gens), "miss": n_miss, "center_dist": center_d,
               "layers": stats, "best_layer": best["layer"]}, open(a.out, "w"),
              indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
