# Winsp - Navigation-Error Recovery: Nanometre-Scale Site Re-localization

A learned localization system for recovering the exact inspection site in a lower-magnification SEM search image from a high-magnification reference image.

## Contents

1. [Problem](#problem)
2. [Contributions](#contributions)
3. [Approach](#approach)
   1. [Why Feature-Space Correlation?](#why-feature-space-correlation)
   2. [Localization Head](#localization-head)
4. [Synthetic SEM Dataset](#synthetic-sem-dataset)
   1. [SEM Image Formation](#sem-image-formation)
   2. [Edge Response](#edge-response)
   3. [Noise](#noise)
   4. [Geometric Degradation](#geometric-degradation)
   5. [Ground Truth](#ground-truth)
5. [Ambiguity Handling](#ambiguity-handling)
6. [Evaluation](#evaluation)
7. [Inference](#inference)
8. [Repository Structure](#repository-structure)
   1. [Core Modules](#core-modules)
9. [Reproducibility](#reproducibility)
10. [Key Design Decisions](#key-design-decisions)
11. [References](#references)
12. [Technical Summary](#technical-summary)

## Problem

In semiconductor e-beam inspection, returning to the exact physical site is critical for reliable re-inspection. At advanced nodes, structures such as FinFET arrays contain highly repetitive patterns, making localization difficult: multiple regions can produce very similar visual responses even when they correspond to different physical sites.

The task is therefore not simply to find a visually similar patch.

> **Given a 100 × 100 high-magnification reference image, recover its physical location inside a 1000 × 1000 lower-magnification search image.**

The system outputs the predicted search-image coordinate `(x, y)` corresponding to the reference site's center.

## Contributions

Winsp was developed collaboratively, with the two major parts of the system developed independently and integrated into a common site-relocalization pipeline.

### [Priyanshi](https://github.com/kirbx01)

**Synthetic SEM dataset generation and acquisition modelling**

Priyanshi designed and implemented the synthetic data-generation pipeline used for training and evaluating the localization system.

Her work covers the construction of the synthetic FinFET-style scenes, randomized inspection-site placement, paired reference/search generation, ground-truth coordinate generation, and the SEM-inspired image degradations applied to each acquisition.

The dataset pipeline includes:

* Procedural FinFET-like periodic structures and larger-scale layout features
* Randomized inspection-site placement
* Independent reference and search image generation
* Poisson shot noise
* Additive Gaussian detector/readout noise
* Edge-response enhancement
* Gaussian PSF blur
* Rotation and scale perturbations
* Vignetting and additional search-image degradation
* Ground-truth physical and image-space coordinates

A key part of the implementation was preserving the independence of the two simulated acquisitions. Reference and search images therefore receive separate stochastic noise realizations rather than sharing a common noise field.

### [Bhaskarjya Nayananju](https://github.com/cwnorhino)

**Learned localization architecture and inference**

Bhaskarjya developed the learned localization and inference components that operate on the generated SEM observations.

His work covers:

* Shared-weight Siamese CNN feature extraction
* Feature-space correlation
* Valid, unpadded correlation
* Learned correlation calibration
* Sub-grid offset regression
* Backbone-aware coordinate geometry
* Correlation-peak candidate extraction
* Ambiguity handling for repetitive semiconductor structures
* Localization evaluation
* Inference-time optimization

These components turn the generated reference/search pairs into a complete site-relocalization system capable of recovering the physical inspection location rather than simply identifying the most visually similar repeated pattern.

### Combined System

The resulting pipeline connects both contributions:

```text
Synthetic SEM Acquisition
        │
        │  Priyanshi
        ↓
Reference + Search + Ground Truth
        │
        │  Bhaskarjya
        ↓
Siamese Feature Extraction
        ↓
Valid Feature Correlation
        ↓
Candidate Localization
        ↓
Offset Refinement
        ↓
Geometry-Aware Coordinate Mapping
        ↓
Recovered Inspection Site
```

The separation is intentional: the dataset generation establishes a controlled but physically motivated localization problem, while the learned localization system solves that problem from the resulting observations.

## Approach

We use a **Siamese convolutional localization network** rather than classical template matching.

The reference and search images are passed through the same CNN backbone, forcing both observations into a shared feature representation. Localization is then performed in feature space rather than directly comparing raw pixels.

```text
100 × 100 Reference                  1000 × 1000 Search
        │                                     │
        └──────────────┬──────────────────────┘
                       ↓
              Shared CNN Backbone
                (shared weights)
                       ↓
             Feature Representations
                       ↓
          Valid Feature Correlation
                       ↓
                NCC Heatmap
                       │
              ┌────────┴────────┐
              ↓                 ↓
        Match Confidence    Offset Head
                              (dx, dy)
              └────────┬────────┘
                       ↓
                Final (x, y)
```

### Why Feature-Space Correlation?

FinFET structures contain dense periodic fins and repeated gate crossings. Direct image-level similarity can therefore produce multiple strong matches.

The network instead learns feature representations that retain the structural information needed to distinguish the target site from repeated local patterns.

The correlation is computed only on the valid, unpadded feature grid. This preserves the actual correspondence between correlation cells and locations in the search image.

### Localization Head

The model produces two complementary outputs:

1. **Correlation heatmap**

   Each valid feature-grid location receives a similarity score indicating how strongly the reference corresponds to that region of the search image.

2. **Sub-grid offset**

   A lightweight offset head predicts `(dx, dy)` for the winning correlation cell, allowing the final coordinate to be refined beyond the discrete feature-grid resolution.

The inference coordinate is obtained by mapping:

```text
feature-grid cell + predicted offset
                    ↓
          search-image coordinate
                    ↓
                  (x, y)
```

The mapping is derived from the backbone geometry rather than using a fixed empirical scale factor.

## Synthetic SEM Dataset

Training requires paired observations of the same physical site at different magnifications.

The dataset generator renders a larger FinFET-style layout at 10× scale and derives:

* **Reference:** 1000 × 1000 source pixels → 100 × 100 model input
* **Search:** 10000 × 10000 source pixels → 1000 × 1000 search image

The underlying layout contains:

* Dense parallel vertical FinFET fins
* Horizontal gate structures crossing the fins
* Larger-scale isolation and routing structures
* Line-edge and linewidth variation
* Randomized site locations

Each pair records the true physical target location used to generate the reference.

```text
Large synthetic die
        │
        ├──────────────→ 1000 × 1000 Search
        │                  lower magnification
        │
        └── target crop → 100 × 100 Reference
                           high magnification

Ground truth
      ↓
(x, y) in search image
```

### SEM Image Formation

The synthetic images are not generated by applying identical noise to both views.

Reference and search images represent independent physical acquisitions, so every image receives its own stochastic realization.

### Edge Response

Feature boundaries are enhanced using a combination of image gradients and Canny edge response before applying a blurred halo.

```text
Sobel gradient magnitude
          +
     Canny edges
          ↓
     edge response
          ↓
 Gaussian halo
          ↓
 SEM-like edge brightening
```

### Noise

The acquisition model combines:

* Poisson shot noise
* Additive Gaussian detector/readout noise
* Additional search-image degradation

The search image is intentionally noisier than the reference.

Noise is generated from independent RNG streams for the two images, preventing artificial pixel-level correspondence between the pair.

### Geometric Degradation

Reference observations are independently perturbed using:

* Rotation: approximately ±3°
* Scale: approximately 0.96–1.04×
* Gaussian PSF blur
* Mild astigmatic blur variation
* Vignetting
* Additional search-image artifacts

These transformations are applied while retaining the known physical target coordinate as ground truth.

### Ground Truth

Every generated pair stores the physical target location together with the image pair.

Example:

```json
{
  "pair_id": "pair_000",
  "target_x_nm": 3240,
  "target_y_nm": 5170,
  "center_x_wide_px": 374.0,
  "center_y_wide_px": 567.0,
  "rotation_deg": -1.72,
  "scale": 1.018
}
```

This allows localization error to be measured directly:

$$
\text{error} =
\sqrt{
(x_{\text{pred}} - x_{\text{gt}})^2 +
(y_{\text{pred}} - y_{\text{gt}})^2
}
$$

## Ambiguity Handling

Periodic semiconductor layouts can generate several near-identical correlation peaks.

Inference therefore does not blindly select an arbitrary candidate.

The system:

1. Extracts local maxima from the correlation heatmap.
2. Discards weak candidates below a relative confidence threshold.
3. Converts candidate feature-grid locations into image coordinates.
4. Identifies near-tied high-confidence candidates.
5. Uses search-frame center proximity as the final tie-break.

The inference code also reports the number of spatially distinct ambiguous candidates, allowing difficult periodic cases to be measured rather than hidden.

## Evaluation

The self-evaluation pipeline reports:

| Metric                    | Purpose                                             |
| ------------------------- | --------------------------------------------------- |
| Mean localization error   | Overall positional accuracy                         |
| Median localization error | Robust central error                                |
| Success @ tolerance       | Fraction of predictions within the target tolerance |
| Normal-case success       | Performance on ordinary samples                     |
| Hard-case success         | Performance on ambiguous samples                    |
| Mean inference time       | Runtime per localization                            |
| Ambiguous-match fraction  | Frequency of multiple plausible candidates          |

The primary localization metric is the Euclidean error between predicted and ground-truth search coordinates.

## Inference

The inference interface accepts:

* Reference image
* Search image

```text
      ↓
Siamese Localizer
      ↓
Correlation heatmap
      ↓
Candidate extraction
      ↓
Offset refinement
      ↓
(x, y)
```

Inference uses `torch.inference_mode()` and GPU-side local-maxima extraction. For the fixed 100 × 100 / 1000 × 1000 input configuration, cuDNN benchmarking is enabled so convolution kernels can be optimized for the deployment shapes.

## Repository Structure

```text
.
├── model.py
├── geometry.py
├── dataset.py
├── generate_dataset.py
├── inference.py
├── train.py
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── checkpoints/
└── README.md
```

### Core Modules

* **model.py**
  Siamese CNN backbone, feature correlation, and offset-regression head.

* **geometry.py**
  Maps feature-grid coordinates to image coordinates using the actual backbone geometry.

* **dataset.py**
  Builds training samples and localization targets on the valid correlation grid.

* **generate_dataset.py**
  Generates the synthetic FinFET SEM dataset and ground-truth coordinates.

* **inference.py**
  Runs localization, candidate extraction, ambiguity handling, coordinate refinement, and evaluation.

## Reproducibility

Dataset generation is seed-controlled.

```bash
python generate_dataset.py \
    --num-samples 1000 \
    --output-dir ./data/train \
    --seed 42
```

Evaluation:

```bash
python inference.py \
    --test-dir ./data/test \
    --checkpoint ./checkpoints/best.pt \
    --device cuda
```

## Key Design Decisions

### Learned Matching Instead of Raw Template Matching

The system learns a feature representation before correlation, allowing the matching function to be optimized for the localization task rather than relying on raw pixel similarity.

### Valid Correlation Instead of Padded Correlation

Only physically valid feature correspondences are retained. This prevents artificial border regions from becoming part of the localization signal.

### Explicit Coordinate Geometry

Feature-grid coordinates are converted back to image coordinates using the backbone's receptive-field and stride geometry.

### Independent Image Acquisition Noise

Reference and search images never share the same noise realization.

### Confidence Before Proximity

The search-frame center is used only to resolve genuinely near-tied candidates rather than overriding a substantially stronger match.

## References

1. Bertinetto, L., Valmadre, J., Henriques, J. F., Vedaldi, A., and Torr, P. H. S. *Fully-Convolutional Siamese Networks for Object Tracking*. ECCV Workshops, 2016.
2. Lewis, J. P. *Fast Template Matching*. Vision Interface, 1995.
3. Utkin, L. V., Kovalev, M. S., and Kasimov, E. M. *An Explanation Method for Siamese Neural Networks*. arXiv:1911.07702, 2019.
4. Canny, J. *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679–698, 1986.
5. Reimer, L. *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*. Springer.
6. Goldstein, J., Newbury, D. E., Joy, D. C., Lyman, C. E., Echlin, P., Lifshin, E., Sawyer, L., and Michael, J. *Scanning Electron Microscopy and X-Ray Microanalysis*. Springer.
7. EstimateNoiseSEM: *A novel framework for deep learning based noise estimation of scanning electron microscopy images*. Ultramicroscopy, 276, 114192, 2025.
8. *Applications of deep learning-based denoising methodologies for scanning electron microscope images*. Measurement Science and Technology.
9. cwnorhino. *Winsp: Navigation-Error Recovery: Nanometre-Scale Site Re-localization*. GitHub repository.

## Technical Summary

```text
Input
100 × 100 Reference
        +
1000 × 1000 Search
        ↓
Shared CNN
        ↓
Feature-space correlation
        ↓
Valid correlation heatmap
        ↓
Peak candidates
        ↓
Offset refinement
        ↓
Geometry-aware coordinate mapping
        ↓
Final inspection-site coordinate
        ↓
              (x, y)
```

**Objective:** recover the same physical inspection site, not merely the most visually similar repeated pattern.
