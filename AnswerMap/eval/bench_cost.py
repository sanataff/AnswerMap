"""
Efficiency benchmark: seconds per map and peak VRAM per explanation method.
Fills the paper's cost table and makes the scale argument concrete, the probe
is 16 cheap forward passes under any fast attention kernel, while white-box
read-outs need eager attention (O(layers x heads x seq^2) materialised) and
T-MM adds a full backward pass. At 30B this difference decides what is
runnable at all.

Methods and their access class:
  probe      2K=16 forwards, sdpa/flash OK, logits only      (black-box)
  occlusion  1+K^2=65 forwards, sdpa/flash OK, logits only   (black-box)
  attn       1 forward, EAGER attention required             (white-box)
  rollout    1 forward, eager, all-layer matmul chain        (white-box)
  tmm        1 forward + 1 backward, eager                   (white-box)

Usage:
  python -m AnswerMap.eval.bench_cost --data data/refcocog/refcocog.jsonl \
      --images_dir data/refcocog/images --n 20 --cache_dir $CD
  python -m AnswerMap.eval.bench_cost ... --model_name Qwen/Qwen3-VL-30B-A3B-Instruct \
      --groups blackbox            # at 30B, white-box may OOM -- that IS the result
"""
import argparse, json, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--side", type=int, default=512)
    ap.add_argument("--attn_layer", type=int, default=15)
    ap.add_argument("--groups", default="blackbox,whitebox",
                    help="which access classes to time. At large scale run "
                         "blackbox only if whitebox OOMs (report the OOM).")
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", default="runs/bench_cost.json")
    a = ap.parse_args()

    import torch
    from functools import partial
    from AnswerMap.answermap import Config, VLM, probe as probe_op, load_image
    import AnswerMap.baselines as B

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][:a.n]
    samples = [(load_image(os.path.join(a.images_dir, r["image"]), a.side),
                r.get("query") or r.get("question")) for r in rows]

    GROUPS = {
        "blackbox": ("sdpa", [
            ("probe",     2 * a.K,       lambda v, c, im, q: probe_op(v, im, q, c, K=a.K)),
            ("occlusion", a.K * a.K + 1, lambda v, c, im, q: B.occlusion_map(v, im, q, c, K=a.K)),
        ]),
        "whitebox": ("eager", [
            ("attn_best", 1, lambda v, c, im, q: B.raw_attention(v, im, q, c, layer=a.attn_layer)),
            ("rollout",   1, lambda v, c, im, q: B.attention_rollout(v, im, q, c)),
            ("tmm",       1, lambda v, c, im, q: B.chefer_relevancy(v, im, q, c)),
        ]),
    }

    results = []
    for gname in [g.strip() for g in a.groups.split(",") if g.strip()]:
        attn_impl, methods = GROUPS[gname]
        cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                     max_side=a.side, min_pixels=a.min_pixels,
                     attn=attn_impl)
        print(f"\n[bench] loading {a.model_name} with attn={attn_impl}")
        vlm = VLM(cfg)
        for name, queries, fn in methods:
            try:
                im0, q0 = samples[0]
                fn(vlm, cfg, im0, q0)                    # warmup, compile paths
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for im, q in samples:
                    fn(vlm, cfg, im, q)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / len(samples)
                mem = torch.cuda.max_memory_allocated() / 1e9
                results.append({"method": name, "group": gname,
                                "queries_per_map": queries,
                                "s_per_map": round(dt, 3),
                                "peak_vram_gb": round(mem, 2)})
                print(f"  {name:<10} {queries:>4} queries   {dt:7.3f} s/map   "
                      f"peak {mem:6.2f} GB")
            except torch.cuda.OutOfMemoryError:
                results.append({"method": name, "group": gname,
                                "queries_per_map": queries, "s_per_map": None,
                                "peak_vram_gb": None, "oom": True})
                print(f"  {name:<10} OOM -- report it, at this scale the "
                      f"white-box read-out is not runnable")
                torch.cuda.empty_cache()
        del vlm
        torch.cuda.empty_cache()

    print(f"\n=== cost per explanation map ({a.model_name}, {a.n} samples, "
          f"side {a.side}) ===")
    print(f"  {'method':<12}{'queries':>9}{'s / map':>10}{'peak GB':>10}  access")
    for r in results:
        s = "OOM" if r.get("oom") else f"{r['s_per_map']:.3f}"
        m = "--" if r.get("oom") else f"{r['peak_vram_gb']:.2f}"
        acc = "logits only" if r["group"] == "blackbox" else "internals + eager"
        print(f"  {r['method']:<12}{r['queries_per_map']:>9}{s:>10}{m:>10}  {acc}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({"model": a.model_name, "n": a.n, "side": a.side,
               "results": results}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
