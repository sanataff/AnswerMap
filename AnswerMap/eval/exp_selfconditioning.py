"""
Self-conditioning (paper 5.3): probe first, then look again where your own
map points. The top-mass region is cropped from the full-resolution image and
appended to the input as a close-up, before any answer is generated. The same
harness runs the cross-model transfer comparison: a SENDER localises the
evidence for a question, a RECEIVER answers.

Channels (each is one row, all consume the SAME sender map):
  text_transfer  sender describes WHERE to look in one phrase (lossy default)
  map_crop       top-mass region cropped from the FULL-RES image, appended as
                 a close-up (region + zoom)
  map_box        red rectangle drawn around the region on the image (the
                 classic visual prompt, keeps full context, no zoom)
  map_dim        everything outside the region dimmed to 35 percent
                 (spotlight, keeps geometry, kills distractors)
  map_text       the map serialized as text, normalized region coordinates
                 plus confidence (the full-map-as-numbers channel)

Controls:
  receiver_alone  no transfer floor
  self_map        the RECEIVER's own map through the same channel (--self_via),
                  does the CROSS-model belief add anything over self-prompting
  random_crop     random region, same channel mechanics (does ANY crop help)
  sender_alone    the sender answering directly (reference)

Usage (the paper run, self rows + cross-model rows in one pass):
  python -m AnswerMap.eval.exp_selfconditioning --data data/textvqa/data.jsonl \
      --images_dir data/textvqa/images \
      --sender Qwen/Qwen3-VL-30B-A3B-Instruct \
      --receiver Qwen/Qwen3-VL-4B-Instruct \
      --cache_dir $CD --limit 500 --out runs/selfcond_textvqa.json
"""
import argparse, json, os, re
import numpy as np
from PIL import Image, ImageDraw


def _vqa_ok(pred, answers):
    def n(s):
        return " ".join(re.sub(r"[^\w\s]", "", (s or "").lower()).split())
    p = n(pred)
    if not p:
        return 0
    gold = [n(x) for x in (answers if isinstance(answers, list) else [answers])]
    return int(any(g and (g == p or g in p or p in g) for g in gold))


HINT_ASK = ("Where in the image is the information needed to answer the "
            "question below? Reply with ONE short location phrase, for example "
            "'bottom left, on the license plate'. Do not answer the question "
            "itself.\nQuestion: {q}")
ANSWER = "{q}\nAnswer with a single word or phrase."
ANSWER_HINT = "{q}\nHint: look at {hint}.\nAnswer with a single word or phrase."
ANSWER_CROP = ("{q}\nThe second image is a close-up of the region most "
               "relevant to the question.\nAnswer with a single word or phrase.")
ANSWER_BOX = ("{q}\nThe red box marks the region most relevant to the "
              "question.\nAnswer with a single word or phrase.")
ANSWER_DIM = ("{q}\nThe bright region is the most relevant to the question.\n"
              "Answer with a single word or phrase.")
ANSWER_MAPTEXT = ("{q}\nA localization system reports the relevant region: "
                  "{maptext}.\nAnswer with a single word or phrase.")


def bbox_of_map(img, M, top_frac=0.12, margin=0.18, min_px=96):
    """bounding box (pixels) of the map's top-mass cells, expanded by margin."""
    K = M.shape[0]
    n_keep = max(1, int(round(top_frac * K * K)))
    idx = np.argsort(M, axis=None)[::-1][:n_keep]
    rr, cc = np.unravel_index(idx, M.shape)
    W, H = img.size
    x0, x1 = cc.min() / K * W, (cc.max() + 1) / K * W
    y0, y1 = rr.min() / K * H, (rr.max() + 1) / K * H
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(W, x1 + mx), min(H, y1 + my)
    if x1 - x0 < min_px:
        cx = (x0 + x1) / 2; x0, x1 = max(0, cx - min_px / 2), min(W, cx + min_px / 2)
    if y1 - y0 < min_px:
        cy = (y0 + y1) / 2; y0, y1 = max(0, cy - min_px / 2), min(H, cy + min_px / 2)
    return (int(x0), int(y0), int(x1), int(y1))


def draw_box(img, bb):
    out = img.copy()
    d = ImageDraw.Draw(out)
    w = max(3, img.size[0] // 250)
    d.rectangle(bb, outline=(220, 40, 40), width=w)
    return out


def dim_outside(img, bb):
    arr = (np.asarray(img).astype(np.float32) * 0.35).astype(np.uint8)
    out = Image.fromarray(arr)
    out.paste(img.crop(bb), bb[:2])
    return out


def maptext_of(img, bb, peak):
    W, H = img.size
    return (f"x from {bb[0]/W:.2f} to {bb[2]/W:.2f}, "
            f"y from {bb[1]/H:.2f} to {bb[3]/H:.2f} in normalized coordinates, "
            f"confidence {peak:.2f}")


def rand_bbox(img, frac=0.12, margin=0.18, seed=0):
    rng = np.random.default_rng(seed)
    W, H = img.size
    side = float(np.sqrt(frac * (1 + 2 * margin) ** 2))
    cw, ch = int(W * side), int(H * side)
    x0 = int(rng.integers(0, max(1, W - cw)))
    y0 = int(rng.integers(0, max(1, H - ch)))
    return (x0, y0, x0 + cw, y0 + ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--sender", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--receiver", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--rows", default="receiver_alone,text_transfer,map_crop,"
                                      "map_box,map_dim,map_text,self_map,"
                                      "random_crop,sender_alone")
    ap.add_argument("--self_via", default="crop", choices=["crop", "box", "dim"],
                    help="which channel the self-map control uses, match it to "
                         "the winning sender channel for the paper run")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--top_frac", type=float, default=0.12)
    ap.add_argument("--probe_side", type=int, default=512)
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="runs/selfcond.json")
    a = ap.parse_args()

    from AnswerMap.answermap import Config, VLM, probe as probe_op, load_image
    rows_wanted = [r.strip() for r in a.rows.split(",") if r.strip()]
    MAP_ROWS = {"map_crop", "map_box", "map_dim", "map_text"}

    cfgS = Config(model_name=a.sender, cache_dir=a.cache_dir, K=a.K,
                  max_side=a.max_side, min_pixels=a.min_pixels)
    cfgSp = Config(model_name=a.sender, cache_dir=a.cache_dir, K=a.K,
                   max_side=a.probe_side, min_pixels=a.min_pixels)
    cfgR = Config(model_name=a.receiver, cache_dir=a.cache_dir, K=a.K,
                  max_side=a.max_side, min_pixels=a.min_pixels)
    cfgRp = Config(model_name=a.receiver, cache_dir=a.cache_dir, K=a.K,
                   max_side=a.probe_side, min_pixels=a.min_pixels)
    print(f"[transfer] sender={a.sender}  receiver={a.receiver}")
    S = VLM(cfgS)
    R = VLM(cfgR)

    data = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        data = data[:a.limit]

    acc = {r: [] for r in rows_wanted}
    out_rows = []
    for i, r in enumerate(data):
        path = os.path.join(a.images_dir, r["image"])
        big = load_image(path, a.max_side)
        small = load_image(path, a.probe_side)
        q = r["question"]; gold = r.get("answers") or r.get("answer")
        gl = gold if isinstance(gold, list) else [gold]
        rec = {"image": r["image"],
               # flag yes/no rows so the open-ended subset can be analysed
               # separately (PathVQA is ~half binary, hints matter less there)
               "binary": int(any(str(g).strip().lower() in ("yes", "no")
                                 for g in gl))}

        bbS = peakS = None
        if any(x in rows_wanted for x in MAP_ROWS):
            res = probe_op(S, small, q, cfgSp)
            M = np.outer(np.asarray(res["c_row"]), np.asarray(res["c_col"]))
            bbS = bbox_of_map(big, M, a.top_frac)
            peakS = float(M.max())

        def ask_R(prompt, imgs):
            return _vqa_ok(R.ask(prompt, imgs=imgs, max_new_tokens=24), gold)

        if "receiver_alone" in rows_wanted:
            rec["receiver_alone"] = ask_R(ANSWER.format(q=q), [big])
        if "sender_alone" in rows_wanted:
            rec["sender_alone"] = _vqa_ok(
                S.ask(ANSWER.format(q=q), imgs=[big], max_new_tokens=24), gold)
        if "text_transfer" in rows_wanted:
            hint = (S.ask(HINT_ASK.format(q=q), imgs=[big], max_new_tokens=24)
                    or "").strip().replace("\n", " ")[:120]
            rec["text_transfer"] = ask_R(ANSWER_HINT.format(q=q, hint=hint), [big])
            rec["hint"] = hint
        if "map_crop" in rows_wanted:
            rec["map_crop"] = ask_R(ANSWER_CROP.format(q=q), [big, big.crop(bbS)])
        if "map_box" in rows_wanted:
            rec["map_box"] = ask_R(ANSWER_BOX.format(q=q), [draw_box(big, bbS)])
        if "map_dim" in rows_wanted:
            rec["map_dim"] = ask_R(ANSWER_DIM.format(q=q), [dim_outside(big, bbS)])
        if "map_text" in rows_wanted:
            rec["map_text"] = ask_R(
                ANSWER_MAPTEXT.format(q=q, maptext=maptext_of(big, bbS, peakS)), [big])
        if "self_map" in rows_wanted:
            resR = probe_op(R, small, q, cfgRp)
            MR = np.outer(np.asarray(resR["c_row"]), np.asarray(resR["c_col"]))
            bbR = bbox_of_map(big, MR, a.top_frac)
            if a.self_via == "crop":
                rec["self_map"] = ask_R(ANSWER_CROP.format(q=q), [big, big.crop(bbR)])
            elif a.self_via == "box":
                rec["self_map"] = ask_R(ANSWER_BOX.format(q=q), [draw_box(big, bbR)])
            else:
                rec["self_map"] = ask_R(ANSWER_DIM.format(q=q), [dim_outside(big, bbR)])
        if "random_crop" in rows_wanted:
            bbX = rand_bbox(big, a.top_frac, seed=i)
            rec["random_crop"] = ask_R(ANSWER_CROP.format(q=q), [big, big.crop(bbX)])

        for k in rows_wanted:
            if k in rec:
                acc[k].append(rec[k])
        out_rows.append(rec)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(data)}] " + "  ".join(
                f"{k}={100*np.mean(v):.1f}" for k, v in acc.items() if v), flush=True)

    DISPLAY = {"receiver_alone": "Receiver alone (no transfer)",
               "text_transfer": "Sender -> words -> receiver",
               "map_crop": "Sender map -> crop close-up",
               "map_box": "Sender map -> red box on image",
               "map_dim": "Sender map -> spotlight (dim rest)",
               "map_text": "Sender map -> coordinates as text",
               "self_map": f"Receiver's own map ({'crop' if True else ''}ctrl)",
               "random_crop": "Random crop (control)",
               "sender_alone": "Sender alone (reference)"}
    DISPLAY["self_map"] = f"Receiver's own map via {a.self_via} (control)"
    print(f"\n=== Spatial transfer, VQA accuracy % (n={len(out_rows)}) ===")
    print(f"  sender {a.sender}\n  receiver {a.receiver}\n")
    summary = {}
    for k in rows_wanted:
        if acc[k]:
            summary[k] = round(100 * float(np.mean(acc[k])), 1)
            print(f"  {DISPLAY[k]:<40}{summary[k]:>7.1f}")
    if "receiver_alone" in summary:
        base = np.asarray(acc["receiver_alone"])
        fails = base == 0
        if fails.sum() > 0:
            print(f"\n  on the {int(fails.sum())} receiver-alone FAILURES, "
                  f"fraction fixed by each channel:")
            for k in rows_wanted:
                if k in ("receiver_alone", "sender_alone") or k not in summary:
                    continue
                if len(acc[k]) == len(base):
                    fx = float(np.asarray(acc[k])[fails].mean())
                    print(f"    {DISPLAY[k]:<40}{100*fx:>6.1f}%")
                    summary[f"fix_{k}"] = round(100 * fx, 1)
    print("\nRead the channels against the controls: a sender channel only counts")
    print("if it beats the receiver's own map through the same channel AND the")
    print("random region through the same mechanics.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({"sender": a.sender, "receiver": a.receiver, "self_via": a.self_via,
               "summary": summary, "results": out_rows}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
