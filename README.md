# Winsp -  Navigation-Error Recovery: Nanometre-Scale Site Re-localization

**Status:**  Project initiated. Dataset generator and matching algorithm are in early scaffolding. Architecture style, model approach, and final augmentation parameters are not yet finalized.

## Background

Wafer inspection tools must repeatedly return to the exact same die site thousands of times per day  with nanometre-level accuracy. Motion-stage drift (thermal expansion, vibration, mechanical slack) causes small positional errors to accumulate between visits, so a tool can land several pixels away from its intended target.

Because every die on a wafer repeats the same circuit layout, a mis-landed image can look nearly identical to the correct one. This makes re-localization inside highly periodic structures (e.g. DRAM arrays, FinFET gates) the core challenge is that Applied Materials calls this **Navigation-Error Recovery**.

Classical template matching struggles here because hundreds of visually near-identical features can appear in a single frame. This project explores AI / computer-vision approaches for more robust recovery.

## Problem Statement

Given:
- A **Reference Image**: a high-resolution crop of the target site.
- A **Search Image**: a lower-magnification (~10x wider field of view) image that contains the reference pattern, shrunk ~10x, somewhere inside it.

Produce:
- The **(x, y)** pixel coordinates of the reference pattern's center within the Search Image.
- If multiple plausible matches exist, return the one **closest to the center of the Search Image**.

## Current State

- [x] Basic scaffold script (`generate_dram_die_canvas`, `apply_sem_effects`, `generate_drift_sense_pair`) exists for DRAM-style generation with independent Poisson + Gaussian noise, Sobel-based edge brightening, and beam-PSF blur.
- [ ] Decide: DRAM-style vs. FinFET-style (or support both).
- [ ] Add rotation and scale-variation augmentations (currently missing).
- [ ] Add periodic/ambiguous "hard negative" regions for failure-mode testing.
- [ ] Generate ≥30 randomized self-evaluation pairs.
- [ ] Build/select the localization algorithm (candidates: classical template/feature matching baseline, normalized cross-correlation, deep-learning matcher).
- [ ] Implement evaluation metric(s) (e.g. pixel error vs. ground truth, success@threshold).
- [ ] Collect citations for every augmentation and noise-model choice.
- [ ] Write final presentation/report.

## Planned Repository Structure

```
.
├── README.md
├── data/
│   ├── generator/           # synthetic dataset generation code
│   │   ├── sem_effects.py       # noise, blur, edge-brightening models
│   │   ├── dram_layout.py       # DRAM-style die canvas generator
│   │   ├── finfet_layout.py     # FinFET-style die canvas generator (TBD)
│   │   └── generate_pairs.py    # builds Reference/Search pairs + ground truth
│   └── samples/              # generated self-evaluation image pairs (≥30)
├── src/
│   ├── matching/              # localization algorithm(s)
│   └── evaluate.py            # scoring against ground truth
├── notebooks/                 # exploration / visualization
├── references/                # citation notes for augmentation & noise choices
└── outputs/                   # figures, result plots, metrics
```

## Open Decisions

- **Layout style:** DRAM-style vs. FinFET-style (both are judged equally; not yet chosen).
- **Matching approach:** classical CV baseline vs. learned matcher, or a hybrid.
- **Noise/augmentation parameters:** exact levels for blur, rotation, scaling, and noise still need to be tuned and cited.

## References

Citations for noise models, edge-brightening behavior, and structural parameters will be collected in `references/` as choices are finalized, per the hackathon's citation requirement.

## Notes

This README will be updated as design decisions are made and the generator/algorithm come in our mind. 