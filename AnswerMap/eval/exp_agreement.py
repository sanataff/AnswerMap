"""
Test 1, agreement with the model's own pointing (paper 4.1): every method's
full map is scored at the model's generated point, in one harness.

CLAIM. The probe map is a faithful readout of the model's spatial belief, and the
model's GENERATED point is a sample from that belief -- so g approx E_M[l]. We
verify this with no external ground truth (self-consistency between two of the
model's own output channels), and show:
  (1) agreement is high, and HIGHER when the map is peaked (low entropy): the map
      knows when it is reliable.
  (2) the probe map predicts g better than an ATTENTION map, because the probe is
      query-conditioned and attention is not.

METHOD COMPARISON = swap the upstream map, hold the metric. For each method we
take its map's EXPECTATION and measure distance to the generated point g.

  probe1       our map, single grid K (rank-1 = outer of marginals)
  probeMG      our map, multigrid product (higher rank)
  attn         Localization Heads attention map (needs eager attn)
  random       control

Data: jsonl with {image, query}. Any pointing/grounding set works
(RefCOCOg, RefCOCO+, CAVE);
no GT needed -- the reference is the model's own generation.

Usage:
  python -m AnswerMap.eval.exp_agreement --data data/refcocog/refcocog.jsonl \
      --images_dir data/refcocog/images \
      --methods probe1,probeMG,attn_best,attn_raw,attn_rollout,occlusion,random \
      --Ks 3,5 --attn_layer 15 --cache_dir $CD --limit 0
"""
import argparse, json, os
import numpy as np
from PIL import Image


def map_entropy(M):
    """normalised Shannon entropy of a map, 0 (peaked) .. 1 (flat)."""
    p = np.asarray(M, float).ravel()
    p = np.clip(p - p.min(), 1e-12, None); p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def nss_auc(M, xn, yn):
    """Saliency-canon map-vs-point metrics (Bylinskii et al., TPAMI'18 -- the
    MIT benchmark family), scoring the FULL map with no expectation collapse
    (fair to multimodal maps, unlike a centroid):
      NSS = z-scored map value at the point (chance 0, higher better)
      AUC = percentile rank of that value among all cells = P(map ranks the true
            cell above a random cell) (chance 0.5, higher better)
    (xn, yn) is the point in [0,1] image coords."""
    M = np.asarray(M, float)
    gh, gw = M.shape
    r = min(gh - 1, max(0, int(yn * gh)))
    c = min(gw - 1, max(0, int(xn * gw)))
    v = float(M[r, c])
    nss = (v - float(M.mean())) / (float(M.std()) + 1e-12)
    flat = M.ravel()
    auc = float(((flat < v).sum() + 0.5 * (flat == v).sum()) / flat.size)
    return float(nss), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--methods",
                    default="probe1,probeMG,attn_best,attn_raw,attn_rollout,random",
                    help="main-table set. Add 'attn' (loc-heads per-example "
                         "adaptation) only for the appendix run.")
    ap.add_argument("--attn_layer", type=int, default=15,
                    help="decoder layer for the attn_best row = attention at its "
                         "BEST layer, found by exp_attn_layer_sweep.py (L15 on "
                         "Qwen3-VL-4B/RefCOCOg; mid-stack, matching Kang et al.'s "
                         "localization-head band. The last layer is FastV-starved).")
    ap.add_argument("--Ks", default="2,3,5")
    ap.add_argument("--combine", default="product")
    ap.add_argument("--K", type=int, default=8, help="single-grid K for probe1")
    ap.add_argument("--probe_side", type=int, default=512)
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--coord_space", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/agreement.json")
    a = ap.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    Ks = tuple(int(x) for x in a.Ks.split(","))
    from AnswerMap.answermap import (Config, VLM, probe as probe_op, load_image,
                           map_expectation, multigrid_map)
    import AnswerMap.baselines as B

    # any attention-family method needs eager (sdpa returns no attentions);
    # exact-membership check would miss attn_raw/attn_best/attn_rollout alone.
    # tmm/tmm_last (Chefer relevancy) backprop through attentions, eager too.
    attn_impl = "eager" if any(m.startswith(("attn", "tmm"))
                               for m in methods) else "sdpa"
    cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                 max_side=a.max_side, min_pixels=a.min_pixels)
    cfg.attn = attn_impl
    vlm = VLM(cfg)
    cfgp = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                  max_side=a.probe_side, min_pixels=a.min_pixels)
    cfgp.attn = attn_impl

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    dist = {m: [] for m in methods}
    nss = {m: [] for m in methods}       # full-map metrics, no expectation collapse
    auc = {m: [] for m in methods}
    devE = {m: [] for m in methods}      # map expectation, deviation from center
    gdev = []                            # generation, deviation from center
    cdist = []                           # centre->generation, SAME diag norm as methods
    ent, live = [], list(methods)
    out_rows, n_gen_ok = [], 0
    for i, r in enumerate(rows):
        path = os.path.join(a.images_dir, r["image"])
        big = load_image(path, a.max_side)
        small = load_image(path, a.probe_side)
        q = r.get("query") or r.get("question")
        W, H = big.size
        diag = float(np.hypot(W, H))

        g, _ = B.generate_point(vlm, big, q, cfg, space=a.coord_space)
        if g is None:
            continue                      # no sample from the belief -> skip
        n_gen_ok += 1

        maps = {}
        if "probe1" in live:
            rr = probe_op(vlm, small, q, cfgp)
            maps["probe1"] = np.outer(np.asarray(rr["c_row"]), np.asarray(rr["c_col"]))
        if "probeMG" in live:
            maps["probeMG"] = multigrid_map(vlm, small, q, cfgp, Ks, a.combine)
        from functools import partial
        for name, fn in (("attn", B.loc_heads), ("attn_raw", B.raw_attention),
                         ("attn_best", partial(B.raw_attention, layer=a.attn_layer)),
                         ("attn_rollout", B.attention_rollout),
                         ("tmm", B.chefer_relevancy),
                         ("tmm_last", partial(B.chefer_relevancy, start_layer=-1))):
            if name in live:
                try:
                    _, dbg = fn(vlm, small, q, cfgp, point_mode="centroid")
                    maps[name] = dbg["map"]
                except Exception as e:
                    print(f"[agreement] {name} disabled ({type(e).__name__}); dropping")
                    live.remove(name)
        if "occlusion" in live:
            maps["occlusion"] = B.occlusion_map(vlm, small, q, cfgp, K=a.K)
        if "random" in live:
            maps["random"] = np.random.default_rng(i).random((16, 16))

        rec = {"image": r["image"], "gen": [round(g[0], 1), round(g[1], 1)],
               "wh": [W, H]}
        for m in live:
            e = map_expectation(maps[m], W, H)
            d = float(np.hypot(g[0] - e[0], g[1] - e[1])) / diag
            dist[m].append(d)
            rec[m] = round(d, 4)
            devE[m].append(((e[0] - W / 2) / W, (e[1] - H / 2) / H))    # map dev
            ns, au = nss_auc(maps[m], g[0] / W, g[1] / H)
            nss[m].append(ns); auc[m].append(au)
            # per-row values so tables can be recomputed
            # from the saved run without a GPU rerun
            rec[f"{m}_exp"] = [round(e[0], 1), round(e[1], 1)]
            rec[f"{m}_nss"] = round(ns, 3); rec[f"{m}_auc"] = round(au, 3)
        gdev.append(((g[0] - W / 2) / W, (g[1] - H / 2) / H))           # gen dev
        cdist.append(float(np.hypot(g[0] - W / 2, g[1] - H / 2)) / diag)
        if "probeMG" in maps:
            ent.append(map_entropy(maps["probeMG"]))
        elif "probe1" in maps:
            ent.append(map_entropy(maps["probe1"]))
        out_rows.append(rec)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(rows)}] gen_ok={n_gen_ok}  " +
                  "  ".join(f"{m}:agree={100*(1-np.mean(dist[m])):.1f}"
                            for m in live if dist[m]), flush=True)

    # center baseline: always predict the image centre. MUST use the same diag
    # normalization as the method distances -- per-axis units inflate the floor
    # by ~sqrt(2) and flatter every method (the tell: a random map, whose
    # expectation IS ~the centre, would look like it beats the centre).
    gd = np.asarray(gdev)                                   # (N, 2) gen deviations
    center_dist = float(np.mean(cdist)) if cdist else float("nan")

    # paper-table grouping: the sections ARE the argument (access class).
    # DISPLAY holds publication names; json keys stay stable for comparability.
    GROUPS = [
        ("Answer-space probing (ours; logits only, black-box)",
         ["probe1", "probeMG"]),
        ("White-box read-outs (attention and gradient)",
         ["attn_best", "attn_raw", "attn_rollout", "tmm", "tmm_last", "attn"]),
        ("Black-box perturbation", ["occlusion"]),
        ("Reference floors", ["random"]),
    ]
    DISPLAY = {"probe1": f"Probe map (K={a.K})",
               "probeMG": f"Probe map (multigrid {a.Ks})",
               "attn_best": f"Attention (layer {a.attn_layer})",
               "attn_raw": "Attention (last layer)",
               "attn_rollout": "Attention rollout",
               "tmm": "Relevancy T-MM (all layers)",
               "tmm_last": "Relevancy T-MM (last block)",
               "attn": "Loc-heads (per-ex. adapt.)",
               "occlusion": "Occlusion (65 queries)",
               "random": "Random map"}

    def stats_for(m):
        md = float(np.mean(dist[m]))
        # Pearson r of coordinates (size-normalized; x,y averaged). Shift-invariant,
        # so this equals the old deviation-from-centre correlation.
        de = np.asarray(devE[m]); n = min(len(de), len(gd))
        r_ = (float(np.corrcoef(de[:n, 0], gd[:n, 0])[0, 1]) +
              float(np.corrcoef(de[:n, 1], gd[:n, 1])[0, 1])) / 2 \
            if de[:n].std() > 1e-9 else float("nan")
        return {"nss": round(float(np.mean(nss[m])), 3),
                "auc": round(float(np.mean(auc[m])), 3),
                "mean_dist": round(md, 4), "r": round(r_, 3),
                "beats_center": round(center_dist - md, 4)}

    W_NAME = 28
    print(f"\n=== Agreement with the model's own pointing (n={n_gen_ok}) ===")
    print(f"{'Method':<{W_NAME}}{'NSS':>7}{'AUC':>7}{'Dist.':>8}{'Pear. r':>9}"
          f"{'D vs centre':>13}")
    print(f"{'':<{W_NAME}}{'(hi=good)':>7}{'':>7}{'(lo=good)':>8}{'':>9}{'':>13}")
    summary = {"center": {"mean_dist": round(center_dist, 4)}}
    for title, members in GROUPS:
        rows = [m for m in members if m in live and dist[m]]
        if not rows:
            continue
        print(f"-- {title} " + "-" * max(0, 66 - len(title)))
        for m in rows:
            s = stats_for(m); summary[m] = s
            print(f"{DISPLAY.get(m, m):<{W_NAME}}{s['nss']:>7.2f}{s['auc']:>7.3f}"
                  f"{s['mean_dist']:>8.3f}{s['r']:>9.3f}{s['beats_center']:>+13.4f}")
        if title == "Reference floors":
            print(f"{'Image-centre prior':<{W_NAME}}{'--':>7}{'--':>7}"
                  f"{center_dist:>8.3f}{'--':>9}{'--':>13}")
    print("""
Metrics: NSS and AUC are the standard saliency map-vs-point scores (Bylinskii
et al., TPAMI'18) -- the full map is evaluated at the model's generated point,
with no reduction to a single expectation, so multimodal maps are treated
fairly. Chance level: NSS 0, AUC 0.5 (see the Random-map row). Dist. is the
distance between the map's expectation and the generated point, as a fraction
of the image diagonal; the Image-centre prior is its floor. Pearson r is the
correlation between map-expectation and generated-point coordinates
(size-normalized, x/y averaged); shift-invariant, so centre bias cannot fake it.""")

    # agreement vs map entropy: does the map know when it is reliable?
    if ent and ("probeMG" in dist or "probe1" in dist):
        key = "probeMG" if dist.get("probeMG") else "probe1"
        e = np.asarray(ent); d = np.asarray(dist[key])
        n = min(len(e), len(d)); e, d = e[:n], d[:n]
        rho = float(np.corrcoef(e, d)[0, 1]) if e.std() > 1e-9 else float("nan")
        order = np.argsort(e)
        lo = d[order[:n // 3]].mean(); hi = d[order[-n // 3:]].mean()
        summary["entropy_dist_corr"] = round(rho, 3)
        summary["dist_peaked_vs_flat"] = [round(float(lo), 4), round(float(hi), 4)]
        print(f"\ncorr(map entropy, gen-distance) = {rho:+.3f}")
        print(f"gen-distance: peaked maps {lo:.4f}  vs  flat maps {hi:.4f}")
        print("A positive correlation is the reliability claim: when the map is")
        print("peaked the model's generation lands ON the expectation; when flat,")
        print("it does not -- and the map's own entropy tells you which.")
    print("\nMain-table rows: probe1, probeMG, attn_best, attn_raw, attn_rollout,")
    print("random, centre. 'attn' (loc-heads) = our per-example ADAPTATION of a")
    print("method whose real protocol needs a calibration corpus -> appendix only,")
    print("worded 'consistent with Kang et al.: without calibrated head identity,")
    print("attention is noise' -- never as their method failing.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({"summary": summary, "results": out_rows}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
