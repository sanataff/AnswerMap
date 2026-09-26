"""
Test 2, causal necessity by deletion (paper 4.2): does the model's answer
DEPEND on the region a map points to? Delete that region and watch the answer
break -- more than for a matched-area random region.

WHY CAUSAL, NOT CORRELATIONAL. Probing "is the car here" and answering "what
colour is the car" are separate forward passes; a peaked probe map does not prove
the answer CAME from that region. So we intervene: corrupt the probe-peak region
and re-ask. If the answer flips, the model used that region. If a RANDOM region's
corruption flips it just as often, the probe found nothing special.

  correct_full     answers correctly with the full image (the only rows that can
                   show a drop)
  drop_<method>    of those, fraction that flip to WRONG when THAT method's
                   top-mass region is corrupted (probe, loc_heads, raw_attention,
                   rollout -- all corrupt the same number of cells)
  drop_random      same, corrupting a random region away from every method (ctrl)

CLAIM: drop_probe >> drop_random  (faithful grounding: the region the answer
causally needs), AND drop_probe > drop_{attention}  (control #1: the probe reads
the model's image-dependence better than the model's own attention -- so the
result is not generic saliency, which any attention map would also catch).

SIGNAL FOR "did it localise": MAX P(yes) over strips (the peak), not entropy. A
high peak means the model strongly commits to a region; bin by it and drop_probe
should rise with the peak.

WHAT TO CORRUPT, AND WHERE TO RUN. One K=8 cell is 1/64 of the image -- too little
to move a robust answer, so we corrupt the probe's TOP-MASS REGION (region_frac of
the cells) and blank it, with a matched-area random region as the control. And the
effect is only large where the answer truly lives in a small region: MMBench (MCQ
reasoning) dilutes it; TEXTVQA is the clean testbed -- the answer is text at a
location, so blanking the probe's region destroys the answer iff the probe was
right. Open-ended and MCQ are both handled (detected from the schema).

Data (either):
  MCQ:         {image, question, choices, answer(idx)}
  open-ended:  {image, question, answers:[...]}   <- textvqa, vqav2

Usage (recommended -- TextVQA, the localized-answer testbed):
  python -m AnswerMap.eval.exp_deletion --data data/textvqa/data.jsonl \
      --images_dir data/textvqa/images --cache_dir $CD --limit 400
"""
import argparse, json, os, re
import numpy as np
from PIL import Image, ImageFilter

LETTERS = "ABCDEFGH"


def parse_letter(txt, k):
    if not txt:
        return None
    m = re.search(r"\b([A-H])\b", txt.upper())
    return LETTERS.index(m.group(1)) if (m and LETTERS.index(m.group(1)) < k) else None


def _vqa_ok(pred, answers):
    """open-ended correctness: exact / contains match against annotator answers."""
    def n(s):
        return " ".join(re.sub(r"[^\w\s]", "", (s or "").lower()).split())
    p = n(pred)
    if not p:
        return 0
    gold = [n(x) for x in (answers if isinstance(answers, list) else [answers])]
    return int(any(g and (g == p or g in p or p in g) for g in gold))


def corrupt_cells(img, cells, K, mode="blank"):
    """Destroy the image content of a SET of K-grid cells (the probe's claimed
    region), leaving the rest intact. Corrupting the true region -- not one cell
    -- is what makes the causal effect measurable."""
    out = img.copy()
    W, H = img.size
    cw, ch = W / K, H / K
    if mode != "blank":
        blurred = img.filter(ImageFilter.GaussianBlur(max(6, min(W, H) // 12)))
    for (r, c) in cells:
        x0, y0 = int(c * cw), int(r * ch)
        x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
        if mode == "blank":
            out.paste(Image.new("RGB", (x1 - x0, y1 - y0), (127, 127, 127)), (x0, y0))
        else:
            out.paste(blurred.crop((x0, y0, x1, y1)), (x0, y0))
    return out


def to_K(M, K):
    """average-pool any (h,w) map onto the KxK cell grid, so every method's
    corruption region is chosen at the SAME resolution and covers the SAME area.
    Handles non-divisible sizes (attention grids are rectangular, e.g. 12x16)."""
    M = np.asarray(M, float)
    h, w = M.shape
    if (h, w) == (K, K):
        return M
    ri = np.minimum((np.arange(h) * K) // h, K - 1)
    ci = np.minimum((np.arange(w) * K) // w, K - 1)
    out = np.zeros((K, K)); cnt = np.zeros((K, K))
    for i in range(h):
        for j in range(w):
            out[ri[i], ci[j]] += M[i, j]; cnt[ri[i], ci[j]] += 1
    return out / np.maximum(cnt, 1)


def top_mass_cells(M, n_keep):
    """the n_keep highest cells of the map -- the region the map points to."""
    idx = np.argsort(M, axis=None)[::-1][:max(1, n_keep)]
    return [tuple(int(v) for v in x) for x in np.array(np.unravel_index(idx, M.shape)).T]


def random_cells(K, n, avoid, seed):
    """n random cells disjoint from `avoid` -- the matched-area control."""
    rng = np.random.default_rng(seed)
    av = set(avoid); out = []
    all_cells = [(r, c) for r in range(K) for c in range(K) if (r, c) not in av]
    rng.shuffle(all_cells)
    return all_cells[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--corrupt", default="blank", choices=["blur", "blank"])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--region_frac", type=float, default=0.12,
                    help="fraction of cells to corrupt = the probe's top-mass "
                         "region (0.12 of 64 cells ~= 8 cells). One cell (1/64) "
                         "is too little to move an answer -- corrupt the region.")
    ap.add_argument("--attn_layer", type=int, default=15,
                    help="layer for raw_attention_best (from exp_attn_layer_sweep: "
                         "L15 on Qwen3-VL-4B -- attention at its best, not a strawman)")
    ap.add_argument("--Ks", default="2,3,5",
                    help="grids for the probeMG method (multigrid product), the "
                         "test-time scaling axis: more coprime grids = more "
                         "independent evidence = sharper map")
    ap.add_argument("--methods", default="probe,loc_heads,raw_attention,raw_attention_best,rollout",
                    help="which upstream maps to corrupt, head-to-head. The "
                         "attention ones are control #1: if the PROBE region drops "
                         "the answer more than the ATTENTION region, the probe reads "
                         "the model's causal dependence better than its own attention "
                         "-- not generic saliency.")
    ap.add_argument("--necessity", default="blank",
                    choices=["blank", "textonly", "off"],
                    help="control #2 -- split the causal drop by whether the image "
                         "is USED for the row. 'blank' (default, clean): does "
                         "blanking the WHOLE image change the answer? Same "
                         "intervention as the region corruption, self-referential, "
                         "no gold/guessing contamination. 'textonly': correct "
                         "without the image (contaminated on yes/no answer spaces -- "
                         "a single open-ended attempt is ~50%% right by chance).")
    ap.add_argument("--exclude_binary", type=int, default=0,
                    help="skip yes/no rows. On binary answers the model's prior is "
                         "right ~50%% of the time, so whole-image blank stays correct "
                         "even when the image WAS used -> the necessity 'unused' label "
                         "gets false negatives. Dropping them gives a clean (smaller) "
                         "image-unused negative control.")
    ap.add_argument("--probe_side", type=int, default=512)
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/deletion.json")
    a = ap.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    ATTN = {"loc_heads", "raw_attention", "raw_attention_best", "rollout", "tmm"}
    # attention baselines need eager attention (sdpa returns no attentions). The
    # probe reads only logits, so eager does not change its result.
    attn = "eager" if any(m in ATTN for m in methods) else "sdpa"

    from AnswerMap.answermap import Config, VLM, probe as probe_op, load_image
    from AnswerMap.baselines import loc_heads, raw_attention, attention_rollout
    cfg = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                 max_side=a.max_side, min_pixels=a.min_pixels, attn=attn)
    vlm = VLM(cfg)
    cfgp = Config(model_name=a.model_name, cache_dir=a.cache_dir, K=a.K,
                  max_side=a.probe_side, min_pixels=a.min_pixels, attn=attn)

    def method_map(name, small, q):
        """each upstream method -> a KxK importance map (peak of the probe map is
        the grounding-strength signal, so it is returned separately)."""
        if name == "probe":
            res = probe_op(vlm, small, q, cfgp)
            M = np.outer(np.asarray(res["c_row"]), np.asarray(res["c_col"]))
            return to_K(M, a.K), float(M.max())
        if name == "probeMG":
            from AnswerMap.answermap import multigrid_map
            Ks = tuple(int(x) for x in a.Ks.split(","))
            M = multigrid_map(vlm, small, q, cfgp, Ks, "product")
            Mk = to_K(M, a.K)
            return Mk, float(Mk.max())
        from functools import partial
        from AnswerMap.baselines import chefer_relevancy
        fn = {"loc_heads": loc_heads, "raw_attention": raw_attention,
              "raw_attention_best": partial(raw_attention, layer=a.attn_layer),
              "rollout": attention_rollout, "tmm": chefer_relevancy}[name]
        _, info = fn(vlm, small, q, cfgp)
        Mk = to_K(info["map"], a.K)
        return Mk, float(Mk.max())

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    def mcq(q, choices):
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        return f"{q}\n{opts}\nAnswer with the letter of the correct option only."

    is_mcq = "choices" in rows[0]
    n_keep = max(1, int(round(a.region_frac * a.K * a.K)))
    print(f"[causal] {'MCQ' if is_mcq else 'open-ended'} data; corrupting "
          f"{n_keep} of {a.K*a.K} cells ({a.corrupt}) = the probe's top-mass region")

    def answer_ok(img, r):
        imgs = None if img is None else [img]      # img=None -> text-only (blind)
        if is_mcq:
            p = mcq(r["question"], r["choices"])
            return int(parse_letter(vlm.ask(p, imgs=imgs, max_new_tokens=8),
                                    len(r["choices"])) == int(r["answer"]))
        p = f"{r['question']}\nAnswer with a single word or phrase."
        return _vqa_ok(vlm.ask(p, imgs=imgs, max_new_tokens=24),
                       r.get("answers") or r.get("answer"))

    n_full = 0
    flips = {m: [] for m in methods}          # per-method: did corrupting ITS region flip?
    flip_rand, probe_peaks, blind = [], [], []
    out_rows = []
    for i, r in enumerate(rows):
        q = r["question"]
        if a.exclude_binary:
            gold = r.get("answers") or r.get("answer") or []
            gold = gold if isinstance(gold, list) else [gold]
            if any(str(g).strip().lower() in ("yes", "no") for g in gold):
                continue                       # binary -> prior contaminates necessity
        path = os.path.join(a.images_dir, r["image"])
        big = load_image(path, a.max_side); small = load_image(path, a.probe_side)

        # only rows correct WITH the full image can show a causal drop
        if not answer_ok(big, r):
            out_rows.append({"image": r["image"], "correct_full": 0}); continue
        n_full += 1
        # control #2: is the image USED for this row? necessity = the same blank
        # intervention at WHOLE-image scale. img_needed=1 iff blanking everything
        # flips the answer to wrong. On img_needed=0 rows the model ignores the
        # image, so corrupting the probe region should NOT flip it (neg control).
        if a.necessity == "blank":
            all_cells = [(rr, cc) for rr in range(a.K) for cc in range(a.K)]
            img_needed = int(not answer_ok(corrupt_cells(big, all_cells, a.K, "blank"), r))
        elif a.necessity == "textonly":
            img_needed = int(not answer_ok(None, r))     # wrong text-only = needed
        else:
            img_needed = -1
        blind.append(img_needed)

        # each method -> its own top-mass region (same area for all: n_keep cells)
        cells = {}
        for m in methods:
            Mk, peak = method_map(m, small, q)
            cells[m] = top_mass_cells(Mk, n_keep)
            if m == "probe":
                probe_peaks.append(peak)
        # one random control per row, drawn AWAY from every method's region
        union = set().union(*[set(c) for c in cells.values()])
        rcells = random_cells(a.K, n_keep, union, seed=i)

        row = {"image": r["image"], "correct_full": 1, "img_needed": img_needed}
        for m in methods:
            f = int(not answer_ok(corrupt_cells(big, cells[m], a.K, a.corrupt), r))
            flips[m].append(f); row[f"flip_{m}"] = f
        fr = int(not answer_ok(corrupt_cells(big, rcells, a.K, a.corrupt), r))
        flip_rand.append(fr); row["flip_random"] = fr
        if probe_peaks:
            row["probe_peak"] = probe_peaks[-1]
        out_rows.append(row)
        if (i + 1) % 20 == 0 and n_full:
            head = "  ".join(f"{m[:5]}={100*np.mean(flips[m]):.0f}%" for m in methods)
            print(f"[{i+1}/{len(rows)}] correct_full={n_full}  {head}  "
                  f"rand={100*np.mean(flip_rand):.0f}%", flush=True)

    DISPLAY = {"probe": "Probe map (ours)",
               "probeMG": f"Probe map (multigrid {a.Ks})",
               "raw_attention": "Attention (last layer)",
               "raw_attention_best": f"Attention (layer {a.attn_layer})",
               "rollout": "Attention rollout",
               "tmm": "Relevancy T-MM (grad x attn)",
               "loc_heads": "Loc-heads (per-ex. adapt.)"}
    print(f"\n=== Deletion test: answer change when a map's region is removed ===")
    print(f"    (n={len(out_rows)}, answered correctly with the full image: {n_full})")
    dr = 100 * np.mean(flip_rand)
    drops = {m: 100 * float(np.mean(flips[m])) for m in methods}
    print(f"  {'Method':<28}{'Answer change (%)':>19}{'D vs random (pts)':>19}")
    print("  " + "-" * 66)
    for m in sorted(methods, key=lambda x: -drops[x]):
        print(f"  {DISPLAY.get(m, m):<28}{drops[m]:>19.1f}{drops[m]-dr:>+19.1f}")
    print(f"  {'Random region (control)':<28}{dr:>19.1f}{0.0:>+19.1f}")
    print("\n  Removing a region (grey blank, matched area) and re-asking: a higher")
    print("  answer-change rate means that method's region is what the answer")
    print("  causally needs. Probe above the attention rows = the probe reads the")
    print("  model's image-dependence better than its own attention (control #1).")

    # does the PROBE peak predict whether corrupting the probe region flips it?
    if "probe" in methods and probe_peaks:
        pk = np.asarray(probe_peaks); fp = np.asarray(flips["probe"])
        if pk.std() > 1e-9:
            order = np.argsort(pk)
            lo = fp[order[:len(pk) // 3]].mean(); hi = fp[order[-len(pk) // 3:]].mean()
            print(f"\n  probe drop when peak LOW: {100*lo:.1f}%  vs HIGH: {100*hi:.1f}%")
            print("  (max P(yes) is the grounding-strength signal -- higher peak,")
            print("   stronger commitment to the region, bigger break when corrupted)")

    # control #2: does corrupting the probe region matter only when the image is
    # USED? Split by whole-image necessity (see the loop). On image-USED rows the
    # probe region should carry most of the necessity; on image-UNUSED rows the
    # probe drop should fall to ~random.
    necessity_report = None
    if a.necessity != "off" and "probe" in methods and blind:
        bl = np.asarray(blind); fp = np.asarray(flips["probe"]); rnd = np.asarray(flip_rand)
        n_used = int((bl == 1).sum()); n_un = int((bl == 0).sum())
        d_used = 100 * float(fp[bl == 1].mean()) if n_used else float("nan")
        d_un = 100 * float(fp[bl == 0].mean()) if n_un else float("nan")
        r_un = 100 * float(rnd[bl == 0].mean()) if n_un else float("nan")
        label = ("whole-image blank flips the answer" if a.necessity == "blank"
                 else "wrong when answered text-only")
        print(f"\n  necessity split ({a.necessity}: image USED = {label}):")
        print(f"    {n_used}/{len(bl)} rows use the image, {n_un} do not")
        print(f"    probe drop on IMAGE-USED rows   : {d_used:.1f}%  (n={n_used})")
        print(f"    probe drop on IMAGE-UNUSED rows : {d_un:.1f}%  (n={n_un})"
              f"   [random here: {r_un:.1f}%]")
        print("    (clean control: on rows the model ignores the image, corrupting")
        print("     the probe region should fall toward random -- and it does iff")
        print("     the map tracks WHEN the image is used, not just where content is)")
        if n_un < 15 or n_used < 15:
            print("    NOTE: a split cell is small here -- read with care.")
        necessity_report = {"mode": a.necessity, "n_used": n_used, "n_unused": n_un,
                            "drop_probe_used": round(d_used, 1),
                            "drop_probe_unused": round(d_un, 1),
                            "drop_random_unused": round(r_un, 1)}
    summary = {"n": len(out_rows), "correct_full": n_full,
               "drop_random": round(dr, 1),
               "drops": {m: round(drops[m], 1) for m in methods},
               "gaps_vs_random": {m: round(drops[m] - dr, 1) for m in methods},
               "necessity_split": necessity_report}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump({"summary": summary, "results": out_rows}, open(a.out, "w"), indent=2)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
