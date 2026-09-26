"""
Coordinate-free pointing (paper 5.2): point-in-region accuracy of every
read-out on a benchmark with gt_box. On a strong pointer the claim is
RECOVERY, the map expectation recovers the model's own generated point. On a
coordinate-blind model (e.g. Lingshu-7B, where a large share of pointing
attempts fail to parse) the expectation converts a probeable spatial belief
into a point with no coordinate emission at all.

Rows: generated point, probe expectation, attention best layer, image centre,
random point. A generation parse-miss is scored as a miss, never dropped.

Data: jsonl {image, query, gt_box:[x0,y0,x1,y1]}  (prep_refcoco.py or
prep_cave.py output).

Usage:
  python -m AnswerMap.eval.exp_pointing --data data/refcocog/refcocog.jsonl \
      --images_dir data/refcocog/images --cache_dir $CD --limit 400
"""
import argparse, json, os
import numpy as np


def in_box(pt, box):
    return int(box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--attn_layer", type=int, default=15)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--probe_side", type=int, default=512)
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/recovery.json")
    a = ap.parse_args()

    from functools import partial
    from AnswerMap.answermap import (Config, VLM, probe as probe_op, load_image,
                           map_expectation)
    import AnswerMap.baselines as B

    cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                 max_side=a.max_side, min_pixels=a.min_pixels,
                 attn="eager")
    vlm = VLM(cfg)
    cfgp = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                  max_side=a.probe_side, min_pixels=a.min_pixels,
                  attn="eager")
    attn_best = partial(B.raw_attention, layer=a.attn_layer)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    hits = {m: [] for m in ("generation", "probe", "attn_best", "center", "random")}
    n_gen_miss = 0
    for i, r in enumerate(rows):
        path = os.path.join(a.images_dir, r["image"])
        big = load_image(path, a.max_side); small = load_image(path, a.probe_side)
        q = r.get("query") or r.get("question")
        W, H = big.size
        # gt_box is in the SAVED file's native pixels (prep_refcoco clipped to
        # the saved image size). The working image is resized to max_side, so
        # rescale by the native-to-working ratio, read from the file itself.
        from PIL import Image as _Image
        ow, oh = _Image.open(path).size
        sx, sy = W / ow, H / oh
        b = [float(v) for v in r["gt_box"]]
        bx = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

        g, _ = B.generate_point(vlm, big, q, cfg)
        if g is None:
            n_gen_miss += 1
            hits["generation"].append(0)             # a miss is a result
        else:
            hits["generation"].append(in_box(g, bx))

        res = probe_op(vlm, small, q, cfgp)
        M = np.outer(np.asarray(res["c_row"]), np.asarray(res["c_col"]))
        e = map_expectation(M, W, H)
        hits["probe"].append(in_box(e, bx))

        try:
            _, info = attn_best(vlm, small, q, cfgp)
            ea = map_expectation(info["map"], W, H)
            hits["attn_best"].append(in_box(ea, bx))
        except Exception as ex:
            print(f"[recovery] attn_best disabled ({type(ex).__name__})")
            hits.pop("attn_best", None)
            attn_best = None

        hits["center"].append(in_box((W / 2, H / 2), bx))
        rng = np.random.default_rng(i)
        hits["random"].append(in_box((rng.uniform(0, W), rng.uniform(0, H)), bx))
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(rows)}] " + "  ".join(
                f"{m}={100*np.mean(v):.1f}" for m, v in hits.items() if v), flush=True)

    print(f"\n=== localization recovery, point-in-box % "
          f"(n={len(hits['probe'])}, generation parse-miss={n_gen_miss}) ===")
    DISPLAY = {"generation": "Generated point (reference)",
               "probe": "Probe expectation (ours)",
               "attn_best": f"Attention (layer {a.attn_layer})",
               "center": "Image-centre prior", "random": "Random point"}
    summary = {}
    for m, v in hits.items():
        if not v:
            continue
        acc = 100 * float(np.mean(v)); summary[m] = round(acc, 1)
        print(f"  {DISPLAY[m]:<30}{acc:>7.1f}")
    print("\nRecovery reading: probe close to generation = the map's expectation")
    print("recovers the model's own pointing without emitting a coordinate.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({"summary": summary, "n": len(hits["probe"])}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
