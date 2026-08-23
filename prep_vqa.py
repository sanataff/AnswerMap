"""
Download and prepare a VQA benchmark into the JSONL the harnesses consume.

    {"image": "000123.jpg", "question": "...", "answers": ["7", "seven", ...]}

Datasets used by the paper:

    textvqa   natural photos containing text. The answer lives at a location,
              so it is the deletion-test (Test 2) and self-conditioning set.
    gqa       compositional questions, the second deletion distribution
              (non-binary subset via --exclude_binary in exp_deletion).
    pope      object-existence probes with yes/no ground truth, feeds
              exp_hallucination.

Field names differ between mirrors and change over time, so the schema is
DETECTED and printed rather than assumed. If detection fails it prints the
available columns and the first example's types instead of guessing.

Usage:
  python prep_vqa.py --dataset textvqa --limit 500 --out_dir data/textvqa \
      --cache_dir $CD
"""
import argparse, json, os

SOURCES = {
    # (hf repo, config, default split)
    "textvqa":        ("lmms-lab/textvqa", None, "validation"),
    # GQA stores Q/A and images in SEPARATE configs; we join them (see below).
    # testdev has public answers (test does not).
    "gqa":            ("lmms-lab/GQA", "testdev_balanced_instructions", "testdev"),
    # POPE: object-hallucination probes, "Is there a X in the image?" with
    # yes/no ground truth over present/absent objects. Feeds exp_hallucination.
    "pope":           ("lmms-lab/POPE", None, "test"),
}
IMAGE_KEYS = ("image", "images", "img", "page_image")
QUESTION_KEYS = ("question", "query", "questions")
ANSWER_KEYS = ("answers", "answer", "gt_answers", "answer_list")


def pick(cols, wanted):
    for w in wanted:
        if w in cols:
            return w
    for c in cols:                       # fall back to a substring match
        if any(w in c.lower() for w in wanted):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="textvqa", choices=sorted(SOURCES))
    ap.add_argument("--repo", default=None, help="override the HF repo id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=500,
                    help="0 = the whole split")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--min_side", type=int, default=0,
                    help="skip images whose longest side is below this")
    a = ap.parse_args()

    repo, cfg, split = SOURCES[a.dataset]
    repo = a.repo or repo
    cfg = a.config if a.config is not None else cfg
    split = a.split or split

    os.environ.setdefault("HF_HOME", a.cache_dir or "")
    from datasets import load_dataset

    if a.dataset == "gqa":
        # GQA: Q/A live in *_instructions (with an imageId), pixels in *_images
        # (id -> image). Join them; the *_images config alone has no questions,
        # which is the DETECTION FAILED you hit.
        instr_cfg = cfg if (cfg and "instruction" in cfg) else "testdev_balanced_instructions"
        img_cfg = instr_cfg.replace("instructions", "images")
        print(f"[prep] GQA join: {instr_cfg} + {img_cfg}  split={split}")
        instr = load_dataset(repo, instr_cfg, split=split, cache_dir=a.cache_dir)
        imgs = load_dataset(repo, img_cfg, split=split, cache_dir=a.cache_dir)
        icols = list(instr.column_names)
        k_iid = "imageId" if "imageId" in icols else pick(icols,
                    ("imageid", "image_id", "imgid", "img_id"))
        if k_iid is None:
            raise SystemExit(f"GQA instructions has no imageId column: {icols}")
        id2idx = {v: i for i, v in enumerate(imgs["id"])}   # id column, no decode
        nn = len(instr) if not a.limit else min(a.limit, len(instr))
        ds = []
        for j in range(nn):
            ex = instr[j]
            k = id2idx.get(ex[k_iid])
            if k is None:
                continue
            ds.append({"image": imgs[k]["image"], "question": ex["question"],
                       "answer": ex["answer"]})
        print(f"[prep] GQA joined {len(ds)} Q/A pairs to images")
        k_img, k_q, k_a = "image", "question", "answer"
        cols = ["image", "question", "answer"]         # writer checks membership
        a.limit = 0                                    # already truncated above
    else:
        print(f"[prep] loading {repo}" + (f" ({cfg})" if cfg else "") + f" split={split}")
        ds = load_dataset(repo, cfg, split=split, cache_dir=a.cache_dir) if cfg else \
            load_dataset(repo, split=split, cache_dir=a.cache_dir)

        cols = list(ds.column_names)
        k_img = pick(cols, IMAGE_KEYS)
        k_q = pick(cols, QUESTION_KEYS)
        k_a = pick(cols, ANSWER_KEYS)
        print(f"[prep] columns: {cols}")
        print(f"[prep] detected  image={k_img!r}  question={k_q!r}  answers={k_a!r}")
        if not (k_img and k_q and k_a):
            ex = ds[0]
            print("\n[prep] DETECTION FAILED. First example, by column:")
            for c in cols:
                v = ex[c]
                print(f"  {c:<24} {type(v).__name__:<12} {str(v)[:60]}")
            raise SystemExit("pass --repo/--config for a mirror with the usual "
                             "column names, or extend IMAGE_KEYS/QUESTION_KEYS/"
                             "ANSWER_KEYS at the top of this file.")

    img_dir = os.path.join(a.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "data.jsonl")

    n = len(ds) if not a.limit else min(a.limit, len(ds))
    kept, skipped_small, sizes = 0, 0, []
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = ds[i]
            im = ex[k_img]
            if isinstance(im, list):                 # some mirrors wrap in a list
                im = im[0]
            if not hasattr(im, "save"):
                continue
            im = im.convert("RGB")
            if a.min_side and max(im.size) < a.min_side:
                skipped_small += 1
                continue
            ans = ex[k_a]
            if isinstance(ans, str):
                ans = [ans]
            ans = [str(x) for x in (ans or []) if str(x).strip()]
            if not ans:
                continue
            name = f"{a.dataset}_{i:06d}.jpg"
            im.save(os.path.join(img_dir, name), quality=95)
            rec = {"image": name, "question": str(ex[k_q]), "answers": ans}
            # pass a category-like column through when one exists (POPE:
            # category, SLAKE: content_type e.g. Position/Organ/Abnormality)
            for ck in ("category", "content_type", "base_type"):
                if ck in cols:
                    rec["category"] = str(ex[ck]); break
            f.write(json.dumps(rec) + "\n")
            sizes.append(max(im.size))
            kept += 1
            if kept % 100 == 0:
                print(f"  {kept} written", flush=True)

    import numpy as np
    print(f"\n[prep] wrote {kept} examples -> {path}")
    print(f"[prep] images -> {img_dir}")
    if skipped_small:
        print(f"[prep] skipped {skipped_small} below --min_side {a.min_side}")
    if sizes:
        s = np.asarray(sizes)
        print(f"[prep] longest side: median {np.median(s):.0f}, "
              f"p10 {np.percentile(s,10):.0f}, p90 {np.percentile(s,90):.0f}")
    print(f"\nnext (Test 2):\n  python exp_deletion.py --data {path} "
          f"--images_dir {img_dir} \\\n"
          f"      --methods probe,raw_attention,raw_attention_best,rollout "
          f"--cache_dir <hf> --limit 400")


if __name__ == "__main__":
    main()
