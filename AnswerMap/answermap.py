"""
The AnswerMap operator: query-conditioned spatial maps read from the
first-token answer posteriors of a frozen vision language model.

    probe(vlm, image, query, cfg)        2K band probes -> K x K map
    multigrid_map(vlm, image, query, .)  product of coprime grids (higher rank)
    map_expectation(M, W, H)             the location read-out, continuous
    M.max()                              the strength read-out, scalar

Black-box by construction: the only access requirement is the first
answer-token logits. No weights, no gradients, no attention. The map exists
before any answer is generated.

THREE THINGS THAT ARE LOAD-BEARING, in case they look arbitrary:

1. A band is fed as ONE reconstructed image strip, not K tile images. The
   processor upscales every small tile to a minimum resolution, so K separate
   tiles cost more to encode than the whole band.
2. The prompt says "ANY of these tiles" even though the model is shown one
   band. The literally-accurate wording scores measurably worse on
   multi-target images. Do not "fix" it.
3. Thinking must be OFF. If the backbone emits a reasoning prefix, the logit
   at position -1 scores that prefix, not the answer, and every probe is
   noise. Support is detected by EFFECT (render the template twice and
   compare), because transformers silently ignores unknown chat-template
   kwargs.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from PIL import Image

YESNO_PROMPT = (
    "You are shown one or more adjacent tiles cropped from a larger image. "
    "Is the following present in ANY of these tiles?\n"
    "\"{cond}\"\n"
    "Answer with exactly one word: Yes or No."
)

GENERATE_PROMPT = ("Point to {q} in this image. Output ONLY the pixel "
                   "coordinates of its center as (x, y). No other text.")


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    cache_dir: Optional[str] = None
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    attn: str = "sdpa"                 # "eager" only for the white-box baselines
    K: int = 8                         # grid side; 2K band probes per map
    max_side: int = 1024
    # THE TWO PROCESSOR FLOORS, which dominate cost and are invisible until
    # measured. Both are in PIXELS (28*28 = 784 px per visual token).
    #   min_pixels: every image is UPSCALED to at least this, so probe cost is
    #     invariant to the render resolution unless you set it.
    #   max_pixels: every image is DOWNSCALED to at most this, so max_side does
    #     NOT control what the model sees on its own.
    # Set both explicitly. None = whatever the checkpoint ships with.
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    probe_batch: int = 4


# --------------------------------------------------------------------------
class VLM:
    """Frozen HF image-text-to-text model. Reads logits; never generates for
    the probe path. Works for any model exposing AutoModelForImageTextToText
    plus a chat template."""

    def __init__(self, cfg: Config):
        import os, logging, warnings, torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # The "processor_kwargs" advisory fires once per forward and buries the
        # output. Cosmetic HERE only because padding was verified applied; do not
        # silence it blind on a new backbone. Suppress it on every transformers
        # logger (parent + children) AND via the warnings module, since it is
        # emitted through both depending on transformers version.
        def _drop(rec):
            return "processor_kwargs" not in rec.getMessage()
        for name in list(logging.root.manager.loggerDict):
            if name.startswith("transformers"):
                logging.getLogger(name).addFilter(_drop)
        logging.getLogger("transformers").addFilter(_drop)
        warnings.filterwarnings("ignore", message=".*processor_kwargs.*")
        self.cfg, self.torch = cfg, torch
        if cfg.cache_dir:
            os.environ.setdefault("HF_HOME", cfg.cache_dir)
        pk = {}
        if cfg.min_pixels is not None:
            pk["min_pixels"] = int(cfg.min_pixels)
        if cfg.max_pixels is not None:
            pk["max_pixels"] = int(cfg.max_pixels)
        try:
            self.processor = AutoProcessor.from_pretrained(
                cfg.model_name, cache_dir=cfg.cache_dir, trust_remote_code=True, **pk)
        except (TypeError, ValueError):
            self.processor = AutoProcessor.from_pretrained(
                cfg.model_name, cache_dir=cfg.cache_dir, trust_remote_code=True)
            pk = {}
        # Some processors accept these only on the image_processor, and some
        # accept them in from_pretrained but ignore them. Set them directly and
        # then REPORT the values in force -- a silently-ignored floor makes probe
        # cost invariant to resolution.
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None:
            for k, v in (("min_pixels", cfg.min_pixels), ("max_pixels", cfg.max_pixels)):
                if v is not None:
                    setattr(ip, k, int(v))
            print(f"[vlm] processor min_pixels={getattr(ip,'min_pixels','?')} "
                  f"max_pixels={getattr(ip,'max_pixels','?')} "
                  f"({getattr(ip,'min_pixels',0)/784:.0f}-"
                  f"{getattr(ip,'max_pixels',0)/784:.0f} visual tokens per image)")
        self.tok = self.processor.tokenizer
        dt = getattr(torch, cfg.dtype)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name, dtype=dt, cache_dir=cfg.cache_dir,
            trust_remote_code=True, attn_implementation=cfg.attn).to(cfg.device)
        self.model.eval()

        # thinking off -- detect by EFFECT, not by absence of an exception
        self.tmpl_kw = {}
        m = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        try:
            base = self.processor.apply_chat_template(m, add_generation_prompt=True,
                                                      tokenize=False)
            alt = self.processor.apply_chat_template(m, add_generation_prompt=True,
                                                     tokenize=False,
                                                     enable_thinking=False)
            if alt != base:
                self.tmpl_kw = {"enable_thinking": False}
                print("[vlm] thinking disabled")
            else:
                print("[vlm] enable_thinking has no effect on this template. Fine for "
                      "a non-thinking model; on a THINKING model every probe would "
                      "score the reasoning prefix -- verify before trusting results.")
        except Exception as e:
            print(f"[vlm] thinking probe failed ({type(e).__name__}); not passing it")

        self.yes_ids = self._first_ids(["Yes", "yes", " Yes", " yes", "YES"])
        self.no_ids = self._first_ids(["No", "no", " No", " no", "NO"])

    def _first_ids(self, strings):
        ids = set()
        for s in strings:
            t = self.tok(s, add_special_tokens=False).input_ids
            if t:
                ids.add(int(t[0]))
        return sorted(ids)

    def _inputs(self, convs):
        return self.processor.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", padding=len(convs) > 1,
            **self.tmpl_kw).to(self.model.device)

    def p_yes_batch(self, img_lists, cond) -> List[float]:
        """P(yes) for each group of images. The 2K probes are independent, so
        they share forward passes. LEFT padding makes logits[:, -1] the last REAL
        token of every row -- with right padding it would read a PAD token."""
        torch = self.torch
        text = YESNO_PROMPT.format(cond=cond)
        old, self.tok.padding_side = getattr(self.tok, "padding_side", "right"), "left"
        yes = torch.tensor(self.yes_ids, device=self.model.device)
        no = torch.tensor(self.no_ids, device=self.model.device)
        out = []
        try:
            B = max(1, self.cfg.probe_batch)
            for i in range(0, len(img_lists), B):
                convs = [[{"role": "user", "content":
                           [{"type": "image", "image": im} for im in imgs] +
                           [{"type": "text", "text": text}]}]
                         for imgs in img_lists[i:i + B]]
                with torch.no_grad():
                    lg = self.model(**self._inputs(convs)).logits[:, -1].float()
                lp = torch.log_softmax(lg, -1)
                p = torch.softmax(torch.stack(
                    [torch.logsumexp(lp[:, yes], -1),
                     torch.logsumexp(lp[:, no], -1)], -1), -1)[:, 0]
                out.extend(p.detach().cpu().tolist())
        finally:
            self.tok.padding_side = old
        return out

    def ask(self, prompt, imgs=None, max_new_tokens=32) -> str:
        torch = self.torch
        conv = [[{"role": "user", "content":
                  [{"type": "image", "image": im} for im in (imgs or [])] +
                  [{"type": "text", "text": prompt}]}]]
        inp = self._inputs(conv)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        return self.tok.decode(out[0, inp["input_ids"].shape[1]:],
                               skip_special_tokens=True)


# --------------------------------------------------------------------------
def load_image(path, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = max_side / max(w, h)
    return im.resize((int(w * s), int(h * s)), Image.LANCZOS) if s < 1.0 else im


def _tiles(img, K):
    W, H = img.size
    tw, th = W // K, H // K
    return [[img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
             for c in range(K)] for r in range(K)]


def _band(tile_list, axis):
    """Paste a strip's tiles back into ONE image. Tiles are contiguous crops, so
    this reconstructs the original band exactly."""
    ws = [t.size[0] for t in tile_list]
    hs = [t.size[1] for t in tile_list]
    if axis == "h":
        out = Image.new("RGB", (sum(ws), max(hs)))
        x = 0
        for t in tile_list:
            out.paste(t, (x, 0)); x += t.size[0]
    else:
        out = Image.new("RGB", (max(ws), sum(hs)))
        y = 0
        for t in tile_list:
            out.paste(t, (0, y)); y += t.size[1]
    return out


def probe(vlm: VLM, image, query: str, cfg: Config, K: Optional[int] = None):
    """The operator. 2K band yes/no probes -> row/column marginals -> rank-1
    K x K map by the outer product -> continuous expectation point."""
    if isinstance(image, str):
        image = load_image(image, cfg.max_side)
    K = K or cfg.K
    W, H = image.size
    t = _tiles(image, K)
    strips = ([[_band([t[r][c] for c in range(K)], "h")] for r in range(K)] +
              [[_band([t[r][c] for r in range(K)], "v")] for c in range(K)])
    print(f"query: {query!r}")
    vals = vlm.p_yes_batch(strips, query)
    c_row, c_col = np.asarray(vals[:K], float), np.asarray(vals[K:2 * K], float)
    M = np.outer(c_row, c_col)

    pr, pc = c_row / (c_row.sum() + 1e-12), c_col / (c_col.sum() + 1e-12)
    er, ec = float((np.arange(K) * pr).sum()), float((np.arange(K) * pc).sum())
    point = ((ec + 0.5) * W / K, (er + 0.5) * H / K)
    return {"image": image, "size": (W, H), "c_row": c_row, "c_col": c_col,
            "M": M, "point": point}


def multigrid_map(vlm, image, query, cfg, Ks=(3, 5), combine="product") -> np.ndarray:
    """The multigrid product. Probe several coarse grids, upsample each map to a
    common resolution, and combine elementwise:

        product -> a product of experts, a soft intersection: a location
                   survives only if EVERY view endorses it (the default)
        mean    -> a soft union
        max     -> the sharpest single view wins, fusion discarded

    Coprime grid sizes share no band boundaries, so the views stay decorrelated;
    nested sizes such as {2,4,8} re-ask implied questions and double-count
    evidence. The result is a higher-rank, non-separable map from purely
    separable probes. Returns M normalised to [0, 1]."""
    img = load_image(image, cfg.max_side) if isinstance(image, str) else image
    base = 64
    acc = None
    for K in Ks:
        # NOTE: K must be passed explicitly -- probe() reads cfg.K otherwise,
        # and a multigrid that silently re-runs one grid is a monotone
        # transform of the single-grid map (identical ranking, identical AUC).
        r = probe(vlm, img, query, cfg, K=int(K))
        M = np.outer(np.asarray(r["c_row"], float), np.asarray(r["c_col"], float))
        M = _upsample2d(M, base, base)
        M = (M - M.min()) / (M.max() - M.min() + 1e-9)
        if acc is None:
            acc = M
        elif combine == "product":
            acc = acc * M
        elif combine == "max":
            acc = np.maximum(acc, M)
        else:                                   # mean
            acc = acc + M
    if combine == "mean":
        acc = acc / len(Ks)
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-9)


def _upsample2d(M, th, tw):
    r = np.minimum(np.arange(th) * M.shape[0] // th, M.shape[0] - 1)
    c = np.minimum(np.arange(tw) * M.shape[1] // tw, M.shape[1] - 1)
    return M[np.ix_(r, c)]


def map_expectation(M, W, H):
    """The location read-out: normalized expectation of a 2D map, in pixels.
    Continuous and sub-cell, unlike an argmax over cells."""
    M = np.asarray(M, float)
    M = np.clip(M - M.min(), 0, None)
    s = M.sum() + 1e-12
    gh, gw = M.shape
    er = (M.sum(1) @ (np.arange(gh) + 0.5)) / s
    ec = (M.sum(0) @ (np.arange(gw) + 0.5)) / s
    return (ec / gw * W, er / gh * H)
