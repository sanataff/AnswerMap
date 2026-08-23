"""
The two baselines that share our frozen backbone, so each table contains its own
mechanism control.

  generate_point  -- ask the SAME model to emit the coordinate. The dispositive
                     baseline: if probing wins here, the difference is the
                     read-out, not the model.
  loc_heads       -- Localization Heads (Kang et al., CVPR'25): select attention
                     heads by spatial entropy, aggregate, derive a region.
                     Reads the model's ATTENTION rather than its ANSWER.

BE GENEROUS WITH THE ATTENTION BASELINE. Its map admits several point read-outs
and the choice moves it by up to 3x (on CAVE: peak 5.4, supra-mean centroid 31.7,
region centre 58.1). Run all three via --loc_point and report its BEST per
benchmark, then say in the paper that you did. Picking one arbitrarily and
getting the low number is not a result.

loc_heads needs attn_implementation="eager"; SDPA/FlashAttention return no
attentions. Set cfg.attn = "eager" before constructing the VLM.
"""
import re
import numpy as np
from PIL import Image

from answermap import load_image, GENERATE_PROMPT


# The backbone's NATIVE grounding prompt and coordinate convention. Getting
# either wrong silently destroys the baseline rather than crashing:
#   * Qwen3-VL emits 0-1000 NORMALISED coordinates. Reading them as pixels cost
#     34-47 points (CAVE 83.8 -> 49.1, Point-Bench 66.1 -> 19.5) and put the
#     baseline BELOW the image-centre floor. A range check cannot detect this:
#     on a 1024px image, 0-1000 values look exactly like pixels.
#   * Using a generic prompt instead of the model's own cost InternVL 29 points
#     on CAVE (60.0 -> 31.0).
# So the convention is DECLARED per backbone, never inferred.
NATIVE = {
    "qwen": dict(
        prompt=('Locate "{q}" in this image and output its center point as JSON: '
                '{{"point_2d": [x, y], "label": "{q}"}}'),
        space="norm1000"),
    "internvl": dict(
        # the documented InternVL grounding template (readthedocs FAQ + repo
        # issues): <ref> wrapping, norm-1000, answer arrives as
        # <box>[[x1,y1,x2,y2]]</box>; parse_point takes the centre. Verified
        # 2026-08-13 after an 84% parse-miss exposed the old paraphrased prompt.
        prompt=("Please provide the bounding box coordinate of the region this "
                "sentence describes: <ref>{q}</ref>"),
        space="norm1000"),
    "generic": dict(
        prompt=("Point to {q} in this image. Output ONLY the pixel coordinates "
                "of its center as (x, y). No other text."),
        space="pixel"),
}


def _family(model_name: str) -> str:
    m = model_name.lower()
    # medical models BUILT ON Qwen-VL inherit Qwen's grounding conventions
    # (Lingshu = Qwen2.5-VL-Instruct base, HuatuoGPT-Vision-Qwen2.5VL likewise)
    if "lingshu" in m or "huatuo" in m:
        return "qwen"
    for k in ("qwen", "internvl"):
        if k in m:
            return k
    return "generic"


def parse_point(txt: str):
    """Every reply format we have seen, in priority order. A bare regex over all
    numbers is not enough: prose numbers ("the 3rd object") outrank the real
    coordinates, and <points x=".." y=".."> replies were scored as misses."""
    if not txt:
        return None
    m = re.search(r'"point_2d"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', txt)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'"point"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', txt)
    if m:                                   # some models emit [y, x] here -- check!
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'<point[s]?\s+x\s*=\s*"(-?[\d.]+)"\s+y\s*=\s*"(-?[\d.]+)"', txt)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'"bbox_2d"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,'
                  r'\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', txt)
    if m:                                   # a box: take its centre
        x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
        return (x0 + x1) / 2, (y0 + y1) / 2
    # InternVL family: <box>[[x1, y1, x2, y2]]</box> or <point>[[x, y]]</point>
    # (norm-1000). The double-bracket list defeats the generic tuple regex
    # below (inner commas), which cost an 84% parse-miss on InternVL3.5-HF.
    m = re.search(r'\[\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)'
                  r'(?:\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+))?\s*\]\]', txt)
    if m:
        if m.group(3) is not None:          # [[x1,y1,x2,y2]] box -> centre
            x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
            return (x0 + x1) / 2, (y0 + y1) / 2
        return float(m.group(1)), float(m.group(2))   # [[x, y]] point
    m = re.search(r'[\[(]\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*[\])]', txt)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def generate_point(vlm, image, query, cfg, space=None, prompt=None):
    """-> (x, y) in working-image pixels, or None if unparseable.

    None must be scored as a MISS, never dropped: 'was asked and did not answer'
    is a result. ALWAYS eyeball the first few raw replies (print the raw txt)
    before trusting a number -- a wrong coordinate space fails silently."""
    if isinstance(image, str):
        image = load_image(image, cfg.max_side)
    nat = NATIVE[_family(cfg.model_name)]
    tmpl = prompt or nat["prompt"]
    space = space or nat["space"]
    txt = vlm.ask(tmpl.format(q=query), imgs=[image]) or ""
    pt = parse_point(txt)
    if pt is None:
        return None, {"raw": txt, "space": space}
    x, y = pt
    W, H = image.size
    if space == "norm1000":
        x, y = x / 1000.0 * W, y / 1000.0 * H
    elif space == "norm1":
        x, y = x * W, y * H
    return (x, y), {"raw": txt, "space": space}


def chefer_relevancy(vlm, image, query, cfg, point_mode="centroid", start_layer=0):
    """T-MM: Generic Attention-model Explainability (Chefer et al., ICCV 2021),
    THE gradient-weighted relevancy baseline of the MLLM-explanation literature
    (used as the relevancy representative by arXiv 2503.13891 and the TAM/LRP
    line). Adapted QUERY-CONDITIONED and ANSWER-FREE to match our comparison
    class: relevance of the yes-logit for 'is {query} present', propagated as
        Abar_l = mean_heads( clamp(grad_l * attn_l, 0) ),  R += Abar_l @ R
    VERIFIED against the official repo (Transformer-MM-Explainability):
      * avg_heads = (grad * cam).clamp(min=0).mean(over heads), NO row
        normalization of Abar (their LXMERT generate_ours and CLIP notebook
        both apply none)
      * R initialized to identity, R = R + Abar @ R, grads/cams detached
      * backprop target = a single class logit (one_hot * output)
      * start_layer: their LXMERT reference uses ALL layers (start_layer=0,
        our default); their CLIP notebook default skips to the LAST block
        only (start_layer=-1 here reproduces that variant) -- run both and
        report the baseline at its best.
    Needs gradients + eager attention (white-box). One forward + one backward."""
    import torch
    from answermap import YESNO_PROMPT
    if isinstance(image, str):
        image = load_image(image, cfg.max_side)
    model, processor = vlm.model, vlm.processor
    text = YESNO_PROMPT.format(cond=query)
    conv = [[{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": text}]}]]
    inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt").to(model.device)
    out = model(**inputs, output_attentions=True)          # grad-enabled forward
    if not out.attentions:
        raise RuntimeError("no attentions; construct the VLM with attn='eager'")
    lp = out.logits[0, -1].float()
    # single class logit, as in their one_hot * output (max over yes variants)
    yes = lp[torch.tensor(vlm.yes_ids, device=lp.device)].max()
    grads = torch.autograd.grad(yes, out.attentions, allow_unused=True)

    ids = inputs["input_ids"][0]
    pos = (ids == _image_token_id(model, processor)).nonzero(as_tuple=True)[0]
    n = pos.numel()
    gthw = inputs.get("image_grid_thw")
    merge = int(getattr(getattr(model.config, "vision_config", model.config),
                        "spatial_merge_size", 2))
    if gthw is not None:
        Hg, Wg = int(gthw[0][1]) // merge, int(gthw[0][2]) // merge
        if Hg * Wg != n:
            g = int(round(float(np.sqrt(n)))); Hg = Wg = g; pos = pos[:g * g]
    else:
        g = int(round(float(np.sqrt(n)))); Hg = Wg = g; pos = pos[:g * g]

    L = out.attentions[0].shape[-1]
    n_layers = len(out.attentions)
    first = (n_layers - 1) if start_layer == -1 else start_layer
    R = torch.eye(L)
    with torch.no_grad():
        for li, (A, G) in enumerate(zip(out.attentions, grads)):
            if li < first or G is None:
                continue
            # official rule: clamp(grad*cam).mean(heads), NO row normalization
            Abar = (G[0] * A[0]).clamp(min=0).mean(0).float().cpu()   # [seq, seq]
            R = R + Abar @ R
    S = R[-1].numpy()[pos.cpu().numpy()].reshape(Hg, Wg)
    Mn = _norm_map(S)
    del out, grads
    return _point_from_map(Mn, image, point_mode), {"map": Mn, "grid": (Hg, Wg)}


def occlusion_map(vlm, image, query, cfg, K=8):
    """Occlusion saliency (Zeiler & Fergus 2014) adapted to a free-text query,
    ANSWER-FREE: importance of cell (i,j) = drop in P(yes | 'is {query} present')
    when that cell is blanked grey. 1 + K^2 forwards (65 for K=8), batched
    through p_yes_batch. Black-box and calibration-free like ours, but ~4x the
    queries of the probe for the same grid and O(K^2) scaling vs our O(K).
    The paper's black-box perturbation baseline."""
    from PIL import Image as _Image
    if isinstance(image, str):
        image = load_image(image, cfg.max_side)
    W, H = image.size
    cw, ch = W / K, H / K
    variants = [image]                       # index 0 = unoccluded base
    for r in range(K):
        for c in range(K):
            v = image.copy()
            x0, y0 = int(c * cw), int(r * ch)
            x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
            v.paste(_Image.new("RGB", (x1 - x0, y1 - y0), (127, 127, 127)), (x0, y0))
            variants.append(v)
    probs = vlm.p_yes_batch([[v] for v in variants], query)
    base = probs[0]
    M = np.maximum(0.0, base - np.asarray(probs[1:], float)).reshape(K, K)
    return _norm_map(M)


def _image_token_id(model, processor):
    for obj in (getattr(model, "config", None), processor):
        for name in ("image_token_id", "image_token_index"):
            v = getattr(obj, name, None)
            if isinstance(v, int):
                return v
    for tok in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>"):
        try:
            i = processor.tokenizer.convert_tokens_to_ids(tok)
            if i is not None and i >= 0:
                return i
        except Exception:
            pass
    raise RuntimeError("could not find the image placeholder token id")


def _grid_and_attn(vlm, image, query, cfg):
    """Shared: run one forward with output_attentions, return
    (attentions, image_token_positions, Hg, Wg). Needs eager attention."""
    import torch
    if isinstance(image, str):
        image = load_image(image, cfg.max_side)
    model, processor = vlm.model, vlm.processor
    conv = [[{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": query}]}]]
    inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    if not out.attentions:
        raise RuntimeError("no attentions; construct the VLM with attn='eager'")
    ids = inputs["input_ids"][0]
    pos = (ids == _image_token_id(model, processor)).nonzero(as_tuple=True)[0]
    n = pos.numel()
    gthw = inputs.get("image_grid_thw")
    merge = int(getattr(getattr(model.config, "vision_config", model.config),
                        "spatial_merge_size", 2))
    if gthw is not None:
        Hg, Wg = int(gthw[0][1]) // merge, int(gthw[0][2]) // merge
        if Hg * Wg != n:
            g = int(round(float(np.sqrt(n)))); Hg = Wg = g; pos = pos[:g * g]
    else:
        g = int(round(float(np.sqrt(n)))); Hg = Wg = g; pos = pos[:g * g]
    return out.attentions, pos, Hg, Wg


def _norm_map(M):
    M = np.asarray(M, float)
    M = M - M.min()
    return M / (M.max() + 1e-12)


def raw_attention(vlm, image, query, cfg, point_mode="centroid", layer=-1):
    """CANONICAL attention-as-explanation: mean over heads, from the final token
    to the image tokens, at `layer` (default last). No head selection, no tricks
    -- exactly what 'the attention map' means. This is the unambiguous attention
    baseline; report it so the anti-attention comparison does not hinge on any
    one paper's head-selection recipe.

    `layer`: FastV (arXiv 2403.06764) shows deep-layer attention to image tokens
    is starved (0.21%% of system-prompt attention efficiency), so the LAST layer
    may be attention at its worst. Sweep layers with exp_attn_layer_sweep.py and
    report attention at its BEST layer -- the generous, unstrawmannable baseline."""
    attns, pos, Hg, Wg = _grid_and_attn(vlm, image, query, cfg)
    A = attns[layer][0, :, -1, :].float().cpu().numpy()   # [heads, seq]
    S = A.mean(0)[pos.cpu().numpy()].reshape(Hg, Wg)      # mean-head, to image
    Mn = _norm_map(S)
    return _point_from_map(Mn, image, point_mode), {"map": Mn, "grid": (Hg, Wg)}


def attention_rollout(vlm, image, query, cfg, point_mode="centroid"):
    """Attention rollout (Abnar & Zuidema 2020): multiply (0.5*A + 0.5*I) across
    layers to account for residual mixing, then read the last token's row over
    image tokens. The standard multi-layer attention attribution."""
    attns, pos, Hg, Wg = _grid_and_attn(vlm, image, query, cfg)
    import torch
    L = attns[0].shape[-1]
    R = np.eye(L, dtype=np.float32)
    for layer in attns:
        A = layer[0].mean(0).float().cpu().numpy()        # [seq, seq] mean-head
        A = 0.5 * A + 0.5 * np.eye(L, dtype=np.float32)
        A = A / (A.sum(-1, keepdims=True) + 1e-9)
        R = A @ R
    S = R[-1][pos.cpu().numpy()].reshape(Hg, Wg)
    Mn = _norm_map(S)
    return _point_from_map(Mn, image, point_mode), {"map": Mn, "grid": (Hg, Wg)}


def _point_from_map(M, image, mode):
    from answermap import load_image
    img = load_image(image, 10**9) if isinstance(image, str) else image
    W, H = img.size
    Hg, Wg = M.shape
    B = np.maximum(M - M.mean(), 0.0)
    if mode == "argmax":
        gy, gx = np.unravel_index(int(np.argmax(M)), M.shape)
        return ((gx + .5) * W / Wg, (gy + .5) * H / Hg)
    Bc = B if B.sum() > 0 else M
    gy = float((Bc.sum(1) @ (np.arange(Hg) + .5)) / (Bc.sum() + 1e-12))
    gx = float((Bc.sum(0) @ (np.arange(Wg) + .5)) / (Bc.sum() + 1e-12))
    return (gx * W / Wg, gy * H / Hg)


def loc_heads(vlm, image, query, cfg, keep=3, sigma=1.0, point_mode="box_centre"):
    """ADAPTATION of Localization Heads (Kang et al., arXiv:2503.06287) to Qwen.
    NOT the authors' exact code -- head selection and the refine loop differ. For
    the faithfulness comparison, report this ALONGSIDE raw_attention and
    attention_rollout so the conclusion does not rest on our adaptation."""
    """Per-image head selection -> combined attention map -> region -> point.

    point_mode: "box_centre" | "centroid" | "argmax"  (see module docstring --
    evaluate all three and report the baseline at its best).
    """
    import torch
    if isinstance(image, str):
        image = load_image(image, cfg.max_side)
    W, H = image.size
    model, processor = vlm.model, vlm.processor

    conv = [[{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": query}]}]]
    inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    if not out.attentions:
        raise RuntimeError("no attentions returned; construct the VLM with attn='eager'")

    ids = inputs["input_ids"][0]
    pos = (ids == _image_token_id(model, processor)).nonzero(as_tuple=True)[0]
    n = pos.numel()
    if n == 0:
        raise RuntimeError("no image tokens found in input_ids")
    # TRUE token grid from image_grid_thw, not sqrt(n): a rectangular image gives
    # a non-square grid (e.g. 192 = 12x16), and sqrt(n) rounds to 14 -> a reshape
    # crash. image_grid_thw is [t, gh_patch, gw_patch]; divide by the merge size.
    gthw = inputs.get("image_grid_thw")
    if gthw is not None:
        merge = int(getattr(getattr(model.config, "vision_config", model.config),
                            "spatial_merge_size", 2))
        Hg, Wg = int(gthw[0][1]) // merge, int(gthw[0][2]) // merge
        if Hg * Wg != n:                 # last resort: fall back to square-crop
            g = int(round(float(np.sqrt(n)))); Hg = Wg = g
            pos = pos[:Hg * Wg]
    else:
        g = int(round(float(np.sqrt(n)))); Hg = Wg = g
        if Hg * Wg != n:
            pos = pos[:Hg * Wg]

    # per-head attention from the last position to the image tokens
    heads = []
    for layer in out.attentions:
        A = layer[0, :, -1, :]           # [n_heads, seq]
        A = A[:, pos].float().cpu().numpy()
        for h in range(A.shape[0]):
            S = A[h].reshape(Hg, Wg)
            p = S / (S.sum() + 1e-12)
            ent = -float((p * np.log(p + 1e-12)).sum())   # spatial entropy
            heads.append((ent, S))
    heads.sort(key=lambda z: z[0])       # lowest entropy = most peaked
    M = np.zeros((Hg, Wg), float)
    for _, S in heads[:keep]:
        M += S
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        M = gaussian_filter(M, sigma=sigma)

    B = np.maximum(M - M.mean(), 0.0)
    ys, xs = np.where(B > 0)
    if xs.size == 0:
        ys, xs = np.where(M >= M.max())
    cw, ch = W / Wg, H / Hg
    box = (xs.min() * cw, ys.min() * ch, (xs.max() + 1) * cw, (ys.max() + 1) * ch)

    if point_mode == "argmax":
        gy, gx = np.unravel_index(int(np.argmax(M)), M.shape)
        pt = ((gx + 0.5) * cw, (gy + 0.5) * ch)
    elif point_mode == "centroid":
        Bc = B if B.sum() > 0 else M
        gy = float((Bc.sum(1) @ (np.arange(Hg) + .5)) / (Bc.sum() + 1e-12))
        gx = float((Bc.sum(0) @ (np.arange(Wg) + .5)) / (Bc.sum() + 1e-12))
        pt = (gx * cw, gy * ch)
    else:                                # box_centre
        pt = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    # normalised attention heatmap, for use as a modulation/importance signal
    Mn = M - M.min()
    Mn = Mn / (Mn.max() + 1e-12)
    return pt, {"box": box, "grid": (Hg, Wg), "map": Mn}
