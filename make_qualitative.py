"""
Render real-data qualitative assets into figures/assets/: map overlays, band
highlights, marginal bars, and deletion pairs. Never draw fake data, every
heatmap, bar, and thumbnail here comes from an actual forward pass.

Two modes:

anchor    one RefCOCOg example -> the operator walk-through assets:
            anchor.png            the image, clean
            anchor_pin.png        + the model's generated point (red pin)
            anchor_band_row.png   one row band highlighted (probe's peak row)
            anchor_band_col.png   one column band highlighted
            marginal_row.png      c_row bar chart (blue, transparent, no axes)
            marginal_col.png      c_col bar chart
            heatmap.png           bare K x K probe map (blue colormap)
            probe_map.png         map overlaid on the image
            map_point.png         overlay + the red pin (the agreement test)
            attn_map_grey.png     best-layer attention map, greyscale, honest
          Picks the example automatically (high peak, off-centre generated
          point, parseable) or takes --index.

deletion  one TextVQA example where deleting the probe region flipped the
          answer and the random region did not -> the deletion-pair assets:
            del_orig.png, del_probe.png, del_random.png
          Rows are located by aligning the causal run json with the data
          jsonl (same order), the probe is re-run to recover the cells.

Usage:
  python make_qualitative.py anchor --data data/refcocog/refcocog.jsonl \
      --images_dir data/refcocog/images --cache_dir $CD --out figures/assets
  python make_qualitative.py deletion --data data/textvqa/data.jsonl \
      --images_dir data/textvqa/images --causal runs/gc_textvqa_tmm.json \
      --cache_dir $CD --out figures/assets
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = (44, 95, 138)        # 2C5F8A, the paper brand
RED = (192, 57, 43)         # C0392B, the pin


def draw_pin(img, xy, r=None):
    out = img.copy()
    d = ImageDraw.Draw(out)
    r = r or max(6, img.size[0] // 60)
    x, y = xy
    d.ellipse([x - r, y - r, x + r, y + r], outline=RED, width=max(3, r // 3))
    d.ellipse([x - r // 4, y - r // 4, x + r // 4, y + r // 4], fill=RED)
    return out


def band_overlay(img, K, index, axis="row"):
    """one band tinted brand blue at 35 percent with a border."""
    out = img.convert("RGBA")
    W, H = img.size
    ov = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if axis == "row":
        y0, y1 = int(index * H / K), int((index + 1) * H / K)
        box = [0, y0, W, y1]
    else:
        x0, x1 = int(index * W / K), int((index + 1) * W / K)
        box = [x0, 0, x1, H]
    d.rectangle(box, fill=BLUE + (90,), outline=BLUE + (255,),
                width=max(2, W // 300))
    return Image.alpha_composite(out, ov).convert("RGB")


def map_overlay(img, M, alpha=0.55, cmap="Blues"):
    """upsample map to the image and alpha-blend a colormap over it."""
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


def save_bar(vals, path, vertical=False):
    """marginal bar chart, brand blue, no axes, transparent background."""
    v = np.asarray(vals, float)
    if vertical:
        fig, ax = plt.subplots(figsize=(0.7, 2.6))
        ax.barh(np.arange(len(v))[::-1], v, color="#2C5F8A")
    else:
        fig, ax = plt.subplots(figsize=(2.6, 0.7))
        ax.bar(np.arange(len(v)), v, color="#2C5F8A")
    ax.axis("off")
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def save_map(M, path, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(np.asarray(M, float), cmap=cmap, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def mode_anchor(a):
    from answermap import Config, VLM, probe as probe_op, load_image
    import baselines as B
    cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                 max_side=a.max_side, min_pixels=a.min_pixels,
                 attn="eager")
    vlm = VLM(cfg)
    cfgp = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                  max_side=a.probe_side, min_pixels=a.min_pixels,
                  attn="eager")
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]

    def build(i):
        r = rows[i]
        path = os.path.join(a.images_dir, r["image"])
        big = load_image(path, a.max_side)
        small = load_image(path, a.probe_side)
        q = r.get("query") or r.get("question")
        g, _ = B.generate_point(vlm, big, q, cfg)
        if g is None:
            return None
        res = probe_op(vlm, small, q, cfgp)
        cr, cc = np.asarray(res["c_row"], float), np.asarray(res["c_col"], float)
        M = np.outer(cr, cc)
        W, H = big.size
        off = np.hypot(g[0] / W - 0.5, g[1] / H - 0.5)
        return dict(r=r, big=big, small=small, q=q, g=g, cr=cr, cc=cc, M=M,
                    peak=float(M.max()), off=off)

    if a.index >= 0:
        best = build(a.index)
        if best is None:
            raise SystemExit(f"row {a.index}: generation unparseable, pick another")
    else:
        best, score = None, -1
        for i in range(min(a.scan, len(rows))):
            c = build(i)
            if c is None:
                continue
            s = c["peak"] * (0.2 + c["off"])     # peaked map + off-centre point
            if s > score:
                best, score = c, s
                best["i"] = i
        if best is None:
            raise SystemExit("no parseable example in the scan window")
        print(f"[anchor] picked row {best['i']}  query='{best['q'][:60]}'  "
              f"peak={best['peak']:.2f} off-centre={best['off']:.2f}")

    os.makedirs(a.out, exist_ok=True)
    big, M, cr, cc, g = best["big"], best["M"], best["cr"], best["cc"], best["g"]
    big.save(os.path.join(a.out, "anchor.png"))
    draw_pin(big, g).save(os.path.join(a.out, "anchor_pin.png"))
    band_overlay(big, a.K, int(np.argmax(cr)), "row").save(
        os.path.join(a.out, "anchor_band_row.png"))
    band_overlay(big, a.K, int(np.argmax(cc)), "col").save(
        os.path.join(a.out, "anchor_band_col.png"))
    save_bar(cc, os.path.join(a.out, "marginal_col.png"), vertical=False)
    save_bar(cr, os.path.join(a.out, "marginal_row.png"), vertical=True)
    save_map(M, os.path.join(a.out, "heatmap.png"))
    map_overlay(big, M).save(os.path.join(a.out, "probe_map.png"))
    draw_pin(map_overlay(big, M), g).save(os.path.join(a.out, "map_point.png"))
    try:
        _, info = B.raw_attention(vlm, best["small"], best["q"], cfgp,
                                  layer=a.attn_layer)
        save_map(info["map"], os.path.join(a.out, "attn_map_grey.png"),
                 cmap="Greys")
    except Exception as e:
        print(f"[anchor] attention asset skipped ({type(e).__name__})")
    print(f"[anchor] 10 assets -> {a.out}  (query for the caption: "
          f"'{best['q']}')")


def mode_deletion(a):
    from answermap import Config, VLM, probe as probe_op, load_image
    from exp_deletion import corrupt_cells, top_mass_cells, random_cells
    causal = json.load(open(a.causal))["results"]
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    n = min(len(causal), len(rows))
    picks = [i for i in range(n)
             if causal[i].get("flip_probe") == 1
             and causal[i].get("flip_random") == 0
             and causal[i].get("image") == rows[i].get("image")]
    if not picks:
        raise SystemExit("no aligned row with flip_probe=1 and flip_random=0")
    i = picks[a.pick % len(picks)]
    r = rows[i]
    print(f"[deletion] row {i}  q='{r['question'][:70]}'  "
          f"answers={r.get('answers') or r.get('answer')}")

    cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                 max_side=a.max_side, min_pixels=a.min_pixels)
    vlm = VLM(cfg)
    cfgp = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                  max_side=a.probe_side, min_pixels=a.min_pixels)
    path = os.path.join(a.images_dir, r["image"])
    big = load_image(path, a.max_side)
    small = load_image(path, a.probe_side)
    res = probe_op(vlm, small, r["question"], cfgp)
    M = np.outer(np.asarray(res["c_row"]), np.asarray(res["c_col"]))
    n_keep = max(1, int(round(a.region_frac * a.K * a.K)))
    pc = top_mass_cells(M, n_keep)
    rc = random_cells(a.K, n_keep, pc, seed=i)

    os.makedirs(a.out, exist_ok=True)
    big.save(os.path.join(a.out, "del_orig.png"))
    corrupt_cells(big, pc, a.K, "blank").save(os.path.join(a.out, "del_probe.png"))
    corrupt_cells(big, rc, a.K, "blank").save(os.path.join(a.out, "del_random.png"))
    print(f"[deletion] 3 assets -> {a.out}  (put question + answers in the "
          f"figure caption; use --pick to browse other candidates, "
          f"{len(picks)} available)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["anchor", "deletion"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--causal", default="runs/gc_textvqa_tmm.json")
    ap.add_argument("--index", type=int, default=-1,
                    help="anchor mode: force a specific row (-1 = auto-pick)")
    ap.add_argument("--pick", type=int, default=0,
                    help="deletion mode: which qualifying example to render")
    ap.add_argument("--scan", type=int, default=60)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--region_frac", type=float, default=0.12)
    ap.add_argument("--attn_layer", type=int, default=15)
    ap.add_argument("--probe_side", type=int, default=512)
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", default="figures/assets")
    a = ap.parse_args()
    (mode_anchor if a.which == "anchor" else mode_deletion)(a)


if __name__ == "__main__":
    main()
