"""
One image, one query, one map. The smallest possible use of the operator.

  python demo.py --image photo.jpg --query "the red car" --out map.png

Prints the row/column yes-posteriors, the expectation point (the location
read-out) and the map maximum (the strength read-out), and saves the map
overlaid on the image. Pass --Ks 3,5 for the multigrid product.
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Ks", default=None,
                    help="comma list, e.g. 3,5 -> multigrid product instead of "
                         "the single K grid")
    ap.add_argument("--probe_side", type=int, default=512)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", default="map.png")
    a = ap.parse_args()

    from answermap import (Config, VLM, probe, multigrid_map, map_expectation,
                           load_image)
    from make_qualitative import map_overlay, draw_pin

    cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                 max_side=a.probe_side, min_pixels=a.min_pixels)
    vlm = VLM(cfg)
    img = load_image(a.image, a.probe_side)
    W, H = img.size

    if a.Ks:
        Ks = tuple(int(x) for x in a.Ks.split(","))
        M = multigrid_map(vlm, img, a.query, cfg, Ks)
        print(f"[demo] multigrid product over grids {Ks}")
    else:
        res = probe(vlm, img, a.query, cfg)
        M = res["M"]
        print("[demo] row P(yes):", np.round(res["c_row"], 2).tolist())
        print("[demo] col P(yes):", np.round(res["c_col"], 2).tolist())

    e = map_expectation(M, W, H)
    print(f"[demo] expectation (location): ({e[0]/W:.2f}, {e[1]/H:.2f}) "
          f"of the image, maximum (strength): {float(np.max(M)):.2f}")
    draw_pin(map_overlay(img, M), e).save(a.out)
    print(f"[demo] saved {a.out}")


if __name__ == "__main__":
    main()
