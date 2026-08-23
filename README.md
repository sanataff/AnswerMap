# AnswerMap: Faithful Black-Box Spatial Interpretability for Vision Language Models
**Authors:** TBD.

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b)](https://arxiv.org/abs/TBD)

</div>

<div align="center">
  <img src="Figures/fig1_teaser.jpg" width="1000">
  <p><em>Two ways to ask a frozen VLM where it looked, judged at the model's own generated point (red pin). Dissecting internals yields an attention map that scores below chance at that point. Asking the sealed model 2K yes/no band questions yields a map that passes both faithfulness tests.</em></p>
</div>


---

## Highlights
- **The AnswerMap operator**: lay K row bands and K column bands over the image and ask the frozen model 2K yes/no questions of the form *"is the {query} in this band"*. The yes posterior is read from **first-token logits**, never from generated text. The outer product of the two marginals gives a K×K query-conditioned map, its **expectation** is a continuous sub-cell location and its **maximum** is a scalar grounding-strength signal.
- **Black-box, training-free, answer-free**: the only access requirement is top-k first-token logprobs, which several closed APIs expose. No weights, no gradients, no attention, and the map exists **before any answer is generated**. An 8×8 map costs 16 forward passes (0.3 to 1.2 s on one A100 from 4B to 30B).
- **A two-test faithfulness protocol**, both tests ground-truth-free and reusable for any spatial explanation method: **agreement** (score the full map at the model's own generated point with the saliency-canon NSS/AUC) and **causal necessity by deletion** (remove the map's top-mass region at matched area against a disjoint random region and re-ask).
- **The probe map passes both tests and the standard tools do not**: NSS 1.36 / AUC 0.824 on RefCOCOg while attention is below chance at *every* decoder layer of the 4B model, including the layer a per-layer sweep hands it. Deleting the map's region flips 53.4% of TextVQA answers against 3.3% for the matched random control. The ordering replicates over three query distributions, four models, and two model families.
- **The multigrid product**: probing several coprime grids and multiplying the maps is a product of experts, a soft intersection of independent coarse views. The {3,5} recipe is a drop-in replacement for the single 8×8 grid at identical 16-query cost (NSS 1.56 / AUC 0.853), and the composed curve is flat-topped across a five-fold budget range where single-grid refinement collapses.
- **What the map buys you**: a label-free **object hallucination audit** on POPE (a low map maximum flags a hallucinated claim at ROC-AUC 0.833), **coordinate-free pointing** for coordinate-blind models (Lingshu-7B points better through its map, 57.7, than through its own generation, 46.7), and **self-conditioning** (crop where your own map points and answer again, fixing half of the model's failures against 38% for a random crop).
- **Plug-and-play**: works out of the box with any HuggingFace `AutoModelForImageTextToText` backbone with a chat template. Evaluated on Qwen3-VL (4B / 8B / 30B-A3B), InternVL3.5-8B, and Lingshu-7B. No fine-tuning, no adapters.
---


## News
- [2026-08] Code released. arXiv preprint TBD.
---


## Methodology
<div align="center">
  <img src="Figures/fig2_method.jpg" width="700">
  <p><em>The AnswerMap operator. K row and K column yes/no band questions fill a K×K map through the outer product, a product of experts over independent coarse views. The crosshair is the expectation read-out, the flag is the maximum. The model stays sealed, first-token logits are the only thing that crosses.</em></p>
</div>

<div align="center">
  <img src="Figures/fig3_tests.jpg" width="700">
  <p><em>The two-test faithfulness protocol. Left, agreement: score the full map at the model's own generated point. Right, deletion: remove the map's top-mass region against a matched random region and re-ask. A self-report can be confabulated, the map is only called an explanation after passing both.</em></p>
</div>

---

# Guide for AnswerMap

## 🧩 Prerequisites
- Python **3.10+**
- CUDA-compatible GPU (all paper numbers were produced on a single **A100 80GB**, bf16)
- `conda` or `pip` package manager

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/TBD/AnswerMap.git
cd AnswerMap

# Install dependencies
pip install -r requirements.txt
```

In the commands below `$CD` is your HuggingFace cache directory (`export CD=/path/to/cache`); drop `--cache_dir $CD` to use the default.

---

## Repository Layout

Every script name matches the paper vocabulary:

| File | Paper section | What it is |
|------|---------------|------------|
| `answermap.py` | Method, the operator | `probe` (2K band probes → K×K map), `multigrid_map` (product of coprime grids), `map_expectation` (location read-out) |
| `baselines.py` | Comparison class | native pointing prompt per family, attention (any layer / rollout / loc-heads), gradient-weighted relevancy T-MM, occlusion |
| `exp_agreement.py` | Test 1, agreement | NSS / AUC / distance / Pearson r of every method's map at the model's own generated point |
| `exp_attn_layer_sweep.py` | Test 1, the fair shot | sweeps every decoder layer so attention enters both tests at its best layer |
| `exp_deletion.py` | Test 2, deletion | answer change rate when each method's top-mass region (12% of cells) is removed, against a matched-area random control; also the image-reliance labelling |
| `exp_hallucination.py` | Application, hallucination audit | POPE yes-claims: deletion signature + the low-peak detector |
| `exp_pointing.py` | Application, coordinate-free pointing | point-in-region accuracy of every read-out on benchmarks with boxes |
| `exp_selfconditioning.py` | Application, self-conditioning | probe → crop → answer, with the random-crop and cross-model controls |
| `bench_cost.py` | Cost per map | seconds and peak VRAM per explanation map, per method, per scale |
| `analyze_runs.py` | Results tables | recompute table numbers from saved run JSONs, no GPU |
| `make_figures.py` | Figures | layer-sweep, expectation-vs-point scatter, query-budget curves |
| `make_qualitative.py` | Qualitative figures | real-data map overlays, band highlights, deletion pairs |
| `prep_refcoco.py`, `prep_vqa.py`, `prep_cave.py` | Setup | download / convert benchmarks into the shared JSONL schema |
| `demo.py` | -- | one image, one query, one map |

---

## Required Data to Reproduce

All prep scripts write the shared schema `{"image", "query"|"question", ...}` plus an `images/` folder.

1. **RefCOCOg / RefCOCO+** (Test 1, sweep, pointing): auto-downloaded from the lmms-lab mirrors.
```bash
python prep_refcoco.py --dataset refcocog --split validation --limit 800 --out_dir data/refcocog --cache_dir $CD
python prep_refcoco.py --dataset refcocop --limit 800 --out_dir data/refcocop --cache_dir $CD
```
2. **TextVQA / GQA / POPE** (Test 2, self-conditioning, hallucination audit): auto-downloaded.
```bash
python prep_vqa.py --dataset textvqa --limit 500 --out_dir data/textvqa --cache_dir $CD
python prep_vqa.py --dataset gqa --limit 800 --out_dir data/gqa --cache_dir $CD
python prep_vqa.py --dataset pope --limit 0 --out_dir data/pope --cache_dir $CD
```
3. **CAVE** (Test 1 on anomaly queries, pointing): no public HF mirror, convert your copy of the [official release](https://github.com/wacv-cave/cave) with `prep_cave.py`.
```bash
python prep_cave.py --src /path/to/cave_annotations.json --images_dir /path/to/cave/images --out_dir data/cave
```

---

## AnswerMap Usage Guide

### Quickstart, one map

```bash
python demo.py --image photo.jpg --query "the red car" --out map.png
```

### Test 1: Agreement with the model's own pointing

First hand attention its best layer (the sweep is one generation plus one forward per example, all layers at once):

```bash
python exp_attn_layer_sweep.py \
  --data data/refcocog/refcocog.jsonl --images_dir data/refcocog/images \
  --limit 200 --cache_dir $CD --out runs/attn_layer_sweep.json
```

Then the full comparison (best layer was 15 on Qwen3-VL-4B, 24 on 8B; re-sweep per model):

```bash
python exp_agreement.py \
  --data data/refcocog/refcocog.jsonl --images_dir data/refcocog/images \
  --methods probe1,probeMG,attn_best,attn_raw,attn_rollout,tmm,tmm_last,occlusion,random \
  --Ks 3,5 --attn_layer 15 \
  --cache_dir $CD --limit 0 --out runs/agreement_refcocog.json
```

Swap `--data`/`--images_dir` to `data/refcocop` or `data/cave` for the other two query distributions. `probe1` is the single 8×8 grid, `probeMG` the multigrid product, `tmm` the gradient-weighted relevancy baseline (needs eager attention and gradients, added automatically).

### Test 2: Causal necessity by deletion

```bash
python exp_deletion.py \
  --data data/textvqa/data.jsonl --images_dir data/textvqa/images \
  --methods probe,raw_attention,raw_attention_best,rollout,tmm --attn_layer 15 \
  --cache_dir $CD --limit 400 --out runs/deletion_textvqa.json
```

Robustness variants from the paper: `--corrupt blur` swaps the grey blank for blur at matched area, and on GQA `--exclude_binary 1` gives the non-binary subset. The whole-image deletion that labels each row by image reliance runs by default (`--necessity blank`).

### Scale and model family

Every harness takes `--model_name`; nothing else changes. The paper grid:

```bash
--model_name Qwen/Qwen3-VL-4B-Instruct      # default, attention best layer 15
--model_name Qwen/Qwen3-VL-8B-Instruct      # re-sweep first, best layer 24
--model_name Qwen/Qwen3-VL-30B-A3B-Instruct
--model_name OpenGVLab/InternVL3_5-8B-HF    # second family, native grounding prompt handled
```

### Application 1: Object hallucination auditing

```bash
python exp_hallucination.py \
  --data data/pope/data.jsonl --images_dir data/pope/images \
  --cache_dir $CD --limit 0 --out runs/halluc_pope.json
```

Reports the deletion signature per claim type and the two detector AUCs (the paper's detector is "AUC low-peak → hallucinated").

### Application 2: Coordinate-free pointing

```bash
python exp_pointing.py \
  --data data/refcocog/refcocog.jsonl --images_dir data/refcocog/images \
  --cache_dir $CD --limit 400 --out runs/pointing_refcocog.json
```

The coordinate-blind row swaps in the medical model (its pointing parse-misses are scored as misses, that is the point):

```bash
python exp_pointing.py \
  --data data/refcocog/refcocog.jsonl --images_dir data/refcocog/images \
  --model_name lingshu-medical-mllm/Lingshu-7B \
  --cache_dir $CD --limit 300 --out runs/pointing_lingshu.json
```

### Application 3: Self-conditioning

One run produces the self-conditioning table (receiver alone / + own map crop / + random crop) and the cross-model channels:

```bash
python exp_selfconditioning.py \
  --data data/textvqa/data.jsonl --images_dir data/textvqa/images \
  --sender Qwen/Qwen3-VL-30B-A3B-Instruct --receiver Qwen/Qwen3-VL-4B-Instruct \
  --self_via crop --cache_dir $CD --limit 500 --out runs/selfcond_textvqa.json
```

A sender channel only counts if it beats the receiver's own map through the same channel AND the random control (in the paper it never does, the practical recipe is self-use).

### Ablations: query budget, fusion rule, cost

The budget and fusion curves are `exp_agreement` runs at different grid settings, all on the same split:

```bash
# refine one grid (single-grid axis)
python exp_agreement.py ... --methods probe1 --K 6  --out runs/agree_K6.json
# compose coprime grids (multigrid axis), and the fusion-rule controls
python exp_agreement.py ... --methods probeMG --Ks 3,5           --out runs/agree_MGfix35.json
python exp_agreement.py ... --methods probeMG --Ks 2,4,8         --out runs/agree_MG248.json
python exp_agreement.py ... --methods probeMG --Ks 2,3,5 --combine mean --out runs/agree_MG235mean.json
```

Cost per explanation map, per access class (at 30B run `--groups blackbox` first; a white-box OOM is itself the result):

```bash
python bench_cost.py \
  --data data/refcocog/refcocog.jsonl --images_dir data/refcocog/images \
  --n 20 --cache_dir $CD --out runs/bench_cost_4b.json
```

### Analysis and Figures (no GPU)

```bash
python analyze_runs.py --agreement runs/agreement_refcocog.json --deletion runs/deletion_textvqa.json
python make_figures.py layers  --in runs/attn_layer_sweep.json  --out figs/
python make_figures.py scatter --in runs/agreement_refcocog.json --out figs/
python make_figures.py budget  --out figs/
python make_qualitative.py anchor --data data/refcocog/refcocog.jsonl --images_dir data/refcocog/images --cache_dir $CD --out figures/assets
```

---

## Configuration Parameters

### Shared Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | JSONL from a prep script | required |
| `--images_dir` | Image folder from the same prep | required |
| `--model_name` | HuggingFace model id | `Qwen/Qwen3-VL-4B-Instruct` |
| `--K` | Grid side (2K band probes per map) | `8` |
| `--Ks` | Multigrid sizes, comma list | `2,3,5` |
| `--probe_side` | Image long side for probing | `512` |
| `--max_side` | Image long side for generation / answering | `1024` |
| `--min_pixels` | Processor floor in pixels (784 px = 1 visual token) | `3136` |
| `--attn_layer` | Decoder layer for the attention-best-layer baseline | `15` |
| `--limit` | Number of rows, `0` = all | varies |
| `--cache_dir` | HuggingFace cache | `None` |
| `--out` | Output JSON path | `runs/*.json` |

### Notable Per-Script Arguments

| Script | Parameter | Description |
|--------|-----------|-------------|
| `exp_agreement.py` | `--methods` | any of `probe1, probeMG, attn_best, attn_raw, attn_rollout, tmm, tmm_last, occlusion, random` (+ `attn` for the loc-heads per-example adaptation) |
| `exp_agreement.py` | `--combine` | multigrid fusion rule: `product` (soft intersection, default), `mean`, `max` |
| `exp_deletion.py` | `--region_frac` | fraction of cells deleted, `0.12` = the top-mass region |
| `exp_deletion.py` | `--corrupt` | `blank` (grey, default) or `blur` at matched area |
| `exp_deletion.py` | `--exclude_binary` | `1` skips yes/no rows (clean image-reliance labels) |
| `exp_deletion.py` | `--necessity` | whole-image deletion for the reliance label, `blank` default |
| `exp_pointing.py` | -- | generation parse-misses are scored as misses, never dropped |
| `exp_selfconditioning.py` | `--rows` | any of `receiver_alone, self_map, random_crop, map_crop, map_box, map_dim, map_text, text_transfer, sender_alone` |
| `exp_selfconditioning.py` | `--self_via` | channel for the receiver's own map: `crop` (default), `box`, `dim` |
| `bench_cost.py` | `--groups` | `blackbox,whitebox`; run `blackbox` alone where white-box OOMs |

---

## 📝 Citation

If you use AnswerMap in your research, please cite:

```bibtex
@misc{TBD2026answermap,
      title={AnswerMap: Faithful Black-Box Spatial Interpretability for Vision Language Models},
      author={TBD},
      year={2026},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/TBD},
}
```
#   A n s w e r M a p  
 