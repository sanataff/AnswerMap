"""
Convert a local CAVE annotation file (EMNLP 2025, 361 reddit-sourced
images, no public HF mirror) into the harness jsonl:

    {"image": "...", "query": "<anomaly description>", "gt_box": [x0,y0,x1,y1]?}

CAVE has no public HF mirror, so this converts your local copy of the
official release; keys are DETECTED (image filename, description, optional
region). On failure it dumps the keys and the first record instead of
guessing, extend the *_KEYS lists with whatever it prints.

Accepts .json (list of records, or dict of id->record) and .jsonl. Regions:
xyxy or xywh boxes, lists of boxes (union bounding box), or polygons (bounding
box of the points). Multiple annotator descriptions: the first non-empty one.

Usage:
  python -m AnswerMap.prep.prep_cave --src /path/to/cave_annotations.json \
      --images_dir /path/to/cave/images --out_dir data/cave

then (Test 1 on anomaly queries, agreement needs only image+query):
  python -m AnswerMap.eval.exp_agreement --data data/cave/cave.jsonl --images_dir data/cave/images \
      --methods probe1,probeMG,attn_best,attn_raw,attn_rollout,occlusion,random \
      --Ks 3,5 --cache_dir $CD --limit 0 --out runs/agreement_cave.json

and if gt_box came through, the coordinate-free pointing table:
  python -m AnswerMap.eval.exp_pointing --data data/cave/cave.jsonl --images_dir data/cave/images \
      --cache_dir $CD --limit 0 --out runs/pointing_cave.json
"""
import argparse, json, os, shutil

IMG_KEYS = ("image", "img", "file", "filename", "file_name", "image_path",
            "img_path", "image_name")
QUERY_KEYS = ("query", "anomaly_description", "description", "anomaly",
              "descriptions", "caption", "text", "expression")
BOX_KEYS = ("gt_box", "box", "bbox", "boxes", "bboxes", "region", "regions",
            "polygon", "polygons", "points")


def pick(keys, wanted):
    for w in wanted:
        if w in keys:
            return w
    low = {k.lower(): k for k in keys}
    for w in wanted:
        for lk, k in low.items():
            if w in lk:
                return k
    return None


def first_text(v):
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)):
        for x in v:
            t = first_text(x)
            if t:
                return t
    if isinstance(v, dict):
        for x in v.values():
            t = first_text(x)
            if t:
                return t
    return ""


def to_box(v):
    """xyxy/xywh box, list of boxes (union), or polygon points -> xyxy or None."""
    if v is None:
        return None
    if isinstance(v, dict):
        for k in ("box", "bbox", "points", "polygon"):
            if k in v:
                return to_box(v[k])
        return None
    if not isinstance(v, (list, tuple)) or not v:
        return None
    flat_nums = all(isinstance(x, (int, float)) for x in v)
    if flat_nums and len(v) == 4:
        x0, y0, a, b = [float(x) for x in v]
        # heuristic identical to prep_refcoco: treat as xywh when the last two
        # read as extents; CAVE-era exports used xyxy, so only convert when
        # a <= x0 or b <= y0 would make xyxy degenerate
        if a > x0 and b > y0:
            return [x0, y0, a, b]                    # already xyxy
        return [x0, y0, x0 + a, y0 + b]              # xywh
    if flat_nums and len(v) % 2 == 0 and len(v) >= 6:   # flat polygon
        xs, ys = v[0::2], v[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    # nested: list of boxes or list of [x,y] points -> union / bounding box
    boxes = [to_box(x) for x in v]
    pts = [x for x in v if isinstance(x, (list, tuple)) and len(x) == 2
           and all(isinstance(c, (int, float)) for c in x)]
    if pts:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    boxes = [b for b in boxes if b]
    if boxes:
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CAVE annotations .json/.jsonl")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", default="data/cave")
    ap.add_argument("--anomalous_only", type=int, default=1,
                    help="skip records with an empty description (the 52 "
                         "normal images have no anomaly to point to)")
    a = ap.parse_args()

    if a.src.endswith(".jsonl"):
        recs = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    else:
        obj = json.load(open(a.src, encoding="utf-8"))
        recs = list(obj.values()) if isinstance(obj, dict) else list(obj)
    if not recs:
        raise SystemExit("no records in --src")

    keys = set()
    for r in recs[:50]:
        if isinstance(r, dict):
            keys.update(r.keys())
    k_img = pick(keys, IMG_KEYS)
    k_q = pick(keys, QUERY_KEYS)
    k_box = pick(keys, BOX_KEYS)
    print(f"[cave] {len(recs)} records, keys: {sorted(keys)}")
    print(f"[cave] detected image={k_img!r} query={k_q!r} box={k_box!r}")
    if not (k_img and k_q):
        print("\n[cave] DETECTION FAILED. First record:")
        print(json.dumps(recs[0], indent=2)[:1200])
        raise SystemExit("extend IMG_KEYS/QUERY_KEYS/BOX_KEYS with the keys above")

    img_out = os.path.join(a.out_dir, "images")
    os.makedirs(img_out, exist_ok=True)
    path = os.path.join(a.out_dir, "cave.jsonl")
    kept, skipped, missing, with_box = 0, 0, 0, 0
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            if not isinstance(r, dict):
                skipped += 1; continue
            name = os.path.basename(str(r.get(k_img, "")))
            q = first_text(r.get(k_q))
            if not name or (a.anomalous_only and not q):
                skipped += 1; continue
            src_img = os.path.join(a.images_dir, name)
            if not os.path.exists(src_img):
                # try common alternates before giving up on the record
                stem = os.path.splitext(name)[0]
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    if os.path.exists(os.path.join(a.images_dir, stem + ext)):
                        name = stem + ext
                        src_img = os.path.join(a.images_dir, name)
                        break
            if not os.path.exists(src_img):
                missing += 1; continue
            dst = os.path.join(img_out, name)
            if not os.path.exists(dst):
                shutil.copy2(src_img, dst)
            rec = {"image": name, "query": q}
            b = to_box(r.get(k_box)) if k_box else None
            if b and b[2] > b[0] and b[3] > b[1]:
                rec["gt_box"] = [round(float(v), 1) for v in b]
                with_box += 1
            f.write(json.dumps(rec) + "\n")
            kept += 1

    print(f"\n[cave] wrote {kept} (skipped {skipped}, images missing {missing}) "
          f"-> {path}")
    print(f"[cave] gt_box present on {with_box}/{kept} "
          f"({'recovery table usable' if with_box else 'agreement only'})")
    print(f"[cave] images -> {img_out}")
    print("\nSPOT-CHECK 3 records visually before trusting boxes (box format was")
    print("auto-guessed). Next: exp_agreement for Test 1, exp_pointing if boxes exist.")


if __name__ == "__main__":
    main()
