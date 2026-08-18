# References

This document records the literature and software sources used to position the Winsp implementation. The references are tied to concrete implementation as referred to in slide deck. Check the presentation for each references.

## 1. Siamese feature correlation

Bertinetto, L., Valmadre, J., Henriques, J. F., Vedaldi, A., and Torr, P. H. S. (2016). *Fully-Convolutional Siamese Networks for Object Tracking*. In European Conference on Computer Vision Workshops, pp. 850–865. DOI: https://doi.org/10.1007/978-3-319-48881-3_56

The main architectural precedent for Winsp. The implementation applies a shared convolutional feature extractor to reference and search images, then performs matching in feature space. The present model uses a four-stage convolutional backbone, L2-normalized feature maps, and valid cross-correlation.

Paper: https://arxiv.org/abs/1606.09549

Project record: https://www.robots.ox.ac.uk/~vgg/publications/2016/BertinettoFC16/

Implementation: https://github.com/cwnorhino/Winsp/blob/main/model.py

## 2. Normalized cross-correlation and template localization

Lewis, J. P. (1995). *Fast Template Matching*. Vision Interface '95, pp. 120–123.

Lewis describes normalized cross-correlation as a similarity measure for locating a template within an image. Winsp does not reproduce classical image-space NCC directly. Instead, the images are embedded by a learned backbone, the feature vectors are L2-normalized, and PyTorch convolution is used as the correlation primitive. The resulting operation is therefore a learned feature-space correlation rather than raw-pixel template matching.

Reference: https://citeseerx.ist.psu.edu/document?doi=57d797a1389ed6211ef39e203eecabcd0d7e37e5&repid=rep1&type=pdf

Implementation: https://github.com/cwnorhino/Winsp/blob/main/model.py

## 3. Siamese representation interpretation

Utkin, L. V., Kovalev, M. S., and Kasimov, E. M. (2019). *An explanation method for Siamese neural networks*. arXiv:1911.07702.

DOI: https://doi.org/10.48550/arXiv.1911.07702

This reference is relevant to the interpretation of decisions made from shared Siamese embedding spaces. It is not cited as the source of Winsp's localization algorithm itself.

Paper: https://arxiv.org/abs/1911.07702

## 4. Edge localization and image formation

Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679–698. DOI: https://doi.org/10.1109/TPAMI.1986.4767851

The connection is to the synthetic SEM formation pipeline, where structural boundaries are enhanced before optical or beam-like blur is introduced. Canny's work provides classical context for the relationship between smoothing, edge response, and localization. Winsp does not use a Canny detector during inference.

Publisher record: https://ieeexplore.ieee.org/document/4767851

Implementation: https://github.com/cwnorhino/Winsp/blob/main/generate_dataset.py

## 5. SEM noise characteristics

Rahman, S. S. M. M., Salomon, M., and Dembélé, S. (2025). *EstimateNoiseSEM: A novel framework for deep learning based noise estimation of scanning electron microscopy images*. Ultramicroscopy, 276, 114192. DOI: https://doi.org/10.1016/j.ultramic.2025.114192

This provides modern context for treating SEM noise as part of the acquisition process rather than as a cosmetic image perturbation. The Winsp dataset generation follows the same general principle by using independent stochastic realizations for reference and search observations and by making the search observation more degraded.

Publisher record: https://www.sciencedirect.com/science/article/pii/S0304399125000907

Implementation: https://github.com/cwnorhino/Winsp/blob/main/generate_dataset.py

## 6. SEM image denoising and measurement context

*Applications of deep learning-based denoising methodologies for scanning electron microscope images*. Measurement Science and Technology. DOI: https://doi.org/10.1088/1361-6501/ad7e41

Useful context for treating SEM noise as part of the measurement process. It supports evaluating localization under degraded observations rather than assuming identical noise in the reference and search images.

Publisher record: https://doi.org/10.1088/1361-6501/ad7e41

## 7. Repository and implementation record

cwnorhino, *Winsp: Navigation-Error Recovery: Nanometre-Scale Site Re-localization*.

Repository: https://github.com/cwnorhino/Winsp

The repository is the authoritative source for implementation-specific claims. The relevant code establishes the following:

- `model.py` defines the shared four-layer CNN with channels 32, 64, 128, 128.
- Feature maps are L2-normalized before correlation.
- Correlation uses an unpadded `conv2d`, preserving only valid reference-in-search correspondences.
- A learned scale and bias calibrate the correlation response.
- A separate convolutional offset head predicts two spatial corrections.
- `geometry.py` derives feature-grid stride and receptive-field origin from the actual convolutional layers.
- `inference.py` extracts local maxima from the response map, converts grid coordinates back into image coordinates, applies offset refinement, resolves near ties using search-frame centre proximity, and records spatially distinct ambiguous candidates.
- The final coordinate is clipped to the search-image bounds.
- Model loading is cached for repeated in-process predictions.

Implementation records:

https://github.com/cwnorhino/Winsp/blob/main/model.py
https://github.com/cwnorhino/Winsp/blob/main/geometry.py
https://github.com/cwnorhino/Winsp/blob/main/inference.py

## 8. Scope of the references

The references are intentionally scoped to mechanisms actually present in the repository. Bertinetto et al. is the primary architectural precedent for shared-weight Siamese matching. Lewis provides the classical correlation background. Utkin et al. concerns Siamese representation interpretation rather than serving as the source of the localization method. Canny provides classical image-processing context for edge treatment. The SEM references provide acquisition and noise context.

Winsp is not presented as a reproduction of any cited method for how the usp of our project is inherently being generated. The implementation combines a learned Siamese feature extractor, valid feature correlation, learned correlation calibration, an offset-regression head, explicit backbone geometry, and ambiguity-aware inference for wafer-inspection re-localization.
