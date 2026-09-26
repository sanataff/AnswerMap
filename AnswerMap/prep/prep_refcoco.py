"""
Download RefCOCO/RefCOCO+/RefCOCOg and emit the jsonl the agreement + localization
harnesses consume:

    {"image": "000123.jpg", "query": "<referring expression>",
     "gt_box": [x0, y0, x1, y1]}

RefCOCOg is the main Test 1 benchmark: its referring
expressions point to VARIED objects all over the image, so the model's generated
point is NOT centre-clustered. We pose it as
"point to <expr>" at eval time; gt_box is kept so the same file also serves the
localization table (point-in-box).

Field names differ across mirrors, so the schema is DETECTED and printed. If
detection fails it dumps the columns and first example instead of guessing.

Usage:
  python -m AnswerMap.prep.prep_refcoco --dataset refcocog --split validation --limit 800 \
      --out_dir data/refcocog --cache_dir $CD
"""
import argparse, json, os

SOURCES = {
    # (hf repo, config, default split). These mirrors carry the image inline and
    # a bbox in xywh or xyxy -- detected below.
    "refcocog": ("lmms-lab/RefCOCOg", None, "validation"),
    "refcoco":  ("lmms-lab/RefCOCO", None, "val"),
    "refcocop": ("lmms-lab/RefCOCOplus", None, "val"),
}
IMG_KEYS = ("image", "img")
EXPR_KEYS = ("answer", "sentence", "sentences", "referring", "expression",
             "caption", "question", "query")
BOX_KEYS = ("bbox", "box", "gt_box", "boxes")


def pick(cols, wanted):
    for w in wanted:
        if w in cols:
            return w
    for c in cols:
        if any(w in c.lower() for w in wanted):
            return c
    return None


def to_xyxy(b):
    """Accept [x,y,w,h] or [x0,y0,x1,y1]; return xyxy. Heuristic: if the 3rd/4th
    look like width/height (smaller than a plausible x1/y1) treat as xywh."""
    b = [float(v) for v in b[:4]]
    x, y, c, d = b
    # xywh if c,d are extents (x+c, y+d plausible) and c<x1-ish -- but ambiguous.
    # RefCOCO mirrors store xywh (COCO). We assume xywh unless it clearly is xyxy
    # (c>x and d>y AND c,d large). Default COCO = xywh.
    return [x, y, x + c, y + d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="refcocog", choices=sorted(SOURCES))
    ap.add_argument("--repo", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--box_format", default="auto", choices=["auto", "xywh", "xyxy"],
                    help="COCO mirrors are xywh; override if a mirror differs")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--min_box_frac", type=float, default=0.0,
                    help="drop boxes below this fraction of the image (tiny/degenerate)")
    a = ap.parse_args()

    repo, cfg, split = SOURCES[a.dataset]
    repo = a.repo or repo
    split = a.split or split
    os.environ.setdefault("HF_HOME", a.cache_dir or "")
    from datasets import load_dataset
    print(f"[prep] loading {repo} split={split}")
    ds = load_dataset(repo, cfg, split=split, cache_dir=a.cache_dir) if cfg else \
        load_dataset(repo, split=split, cache_dir=a.cache_dir)

    cols = list(ds.column_names)
    k_img = pick(cols, IMG_KEYS); k_expr = pick(cols, EXPR_KEYS); k_box = pick(cols, BOX_KEYS)
    print(f"[prep] columns: {cols}")
    print(f"[prep] detected image={k_img!r} expr={k_expr!r} box={k_box!r}")
    if not (k_img and k_expr and k_box):
        ex = ds[0]
        print("\n[prep] DETECTION FAILED. First example:")
        for c in cols:
            print(f"  {c:<22}{type(ex[c]).__name__:<12}{str(ex[c])[:60]}")
        raise SystemExit("pass --repo for a mirror with usual columns, or extend "
                         "the *_KEYS lists.")

    img_dir = os.path.join(a.out_dir, "images"); os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{a.dataset}.jsonl")
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    kept, skipped = 0, 0
    import numpy as np
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = ds[i]
            im = ex[k_img]
            if isinstance(im, list):
                im = im[0]
            if not hasattr(im, "save"):
                skipped += 1; continue
            im = im.convert("RGB"); W, H = im.size

            expr = ex[k_expr]
            if isinstance(expr, list):                 # some store a list of exprs
                expr = expr[0]
            expr = str(expr).strip()

            box = ex[k_box]
            if isinstance(box, list) and box and isinstance(box[0], (list, tuple)):
                box = box[0]                            # list of boxes -> first
            fmt = a.box_format
            if fmt == "auto":
                fmt = "xywh"                            # COCO default
            b = ([float(box[0]), float(box[1]), float(box[0]) + float(box[2]),
                  float(box[1]) + float(box[3])] if fmt == "xywh"
                 else [float(v) for v in box[:4]])
            b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
            if b[2] - b[0] < 2 or b[3] - b[1] < 2 or not expr:
                skipped += 1; continue
            if a.min_box_frac and ((b[2] - b[0]) * (b[3] - b[1])) / (W * H) < a.min_box_frac:
                skipped += 1; continue

            name = f"{a.dataset}_{i:06d}.jpg"
            im.save(os.path.join(img_dir, name), quality=95)
            f.write(json.dumps({"image": name, "query": expr,
                                "gt_box": [round(v, 1) for v in b]}) + "\n")
            kept += 1
            if kept % 100 == 0:
                print(f"  {kept} written", flush=True)

    print(f"\n[prep] wrote {kept} (skipped {skipped}) -> {path}")
    print(f"[prep] images -> {img_dir}")
    print("\nSPOT-CHECK a couple of boxes vs the referring expression before "
          "trusting the box_format (COCO=xywh; if boxes look shifted, try --box_format xyxy).")
    print(f"\nnext (Test 1):\n  python -m AnswerMap.eval.exp_agreement --data {path} "
          f"--images_dir {img_dir} \\\n"
          f"      --methods probe1,probeMG,attn_best,attn_raw,attn_rollout,"
          f"occlusion,random --Ks 3,5 --cache_dir <hf>")


if __name__ == "__main__":
    main()
