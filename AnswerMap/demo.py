"""
One image, one query, one map. The smallest possible use of the operator.

  python -m AnswerMap.demo --image photo.jpg --query "the red car" --out map.png

Prints the row/column yes-posteriors, the expectation point (the location
read-out) and the map maximum (the strength read-out), and saves the map
overlaid on the image. Pass --Ks 3,5 for the multigrid product.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw

RED = (192, 57, 43)


def draw_pin(img, xy, r=None):
    """mark the expectation point with a red pin."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    r = r or max(6, img.size[0] // 60)
    x, y = xy
    d.ellipse([x - r, y - r, x + r, y + r], outline=RED, width=max(3, r // 3))
    d.ellipse([x - r // 4, y - r // 4, x + r // 4, y + r // 4], fill=RED)
    return out


def map_overlay(img, M, alpha=0.55, cmap="Blues"):
    """upsample map to the image and alpha-blend a colormap over it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W, H = img.size
    Mn = np.asarray(M, float)
    Mn = (Mn - Mn.min()) / (Mn.max() - Mn.min() + 1e-12)
    up = np.array(Image.fromarray((Mn * 255).astype(np.uint8))
                  .resize((W, H), Image.BILINEAR)) / 255.0
    rgba = plt.get_cmap(cmap)(up)
    over = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    a = Image.fromarray((up * alpha * 255).astype(np.uint8))
    out = img.copy()
    out.paste(over, (0, 0), a)
    return out


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

    from AnswerMap.answermap import (Config, VLM, probe, multigrid_map, map_expectation,
                           load_image)

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
