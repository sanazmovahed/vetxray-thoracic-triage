# VetXRay Thoracic Radiograph Triage

Can a radiograph classifier tell when it should not be trusted?

This project studies lesion detection on canine and feline thoracic radiographs
using **frozen pretrained backbones**, and evaluates not only accuracy but also
**calibration**, **selective prediction** (abstaining and deferring to a
radiologist) and the effect of **annotated image quality**.

> **Status:** work in progress. No results are reported in this repository until
> they have been produced by a saved run in `results/`.

## Motivation

In a clinical workflow, a model that is right most of the time is not enough.
It also matters whether its confidence can be trusted, and whether it can flag
cases that a human should review. This project asks these questions on a public
veterinary radiograph dataset.

## Dataset

**VetXRay v2** (Zenodo record 19051776, March 2026): 9,882 canine and feline
thoracic radiographs annotated by radiologists with 17 lesion tags and 5
image-quality tags.

The data is **not included** in this repository. Download it from the Zenodo
record and read its license and citation terms there.

Details such as exact column names, image format and whether a patient
identifier exists are established by the exploratory analysis step
(`scripts/01_eda.py`) and documented in `notes/`.

## Research questions

1. **Backbone comparison.** How do two frozen general-purpose backbones
   (a supervised ImageNet model and a self-supervised foundation model) compare
   in lesion detection and in the reliability of their predictions?
2. **Image quality.** How do the annotated quality tags relate to performance
   and to model uncertainty?
3. **Reliability.** Are the predicted probabilities calibrated, and does
   abstaining on uncertain cases give a useful risk-coverage trade-off?

## Approach

1. Inspect the metadata and decide label sets and the splitting strategy.
2. Resize and cache the radiographs once.
3. Extract embeddings once from two frozen general-purpose backbones
   (one pretrained on ImageNet, one a foundation model).
4. Train lightweight heads (linear / small MLP) on the embeddings.
5. Evaluate discrimination, calibration, selective prediction and quality strata.

## Evaluation protocol

- Patient-level split if a patient or study identifier exists; otherwise the
  limitation is stated explicitly.
- Multi-label stratified split with fixed seeds; at least three seeds, reported
  as mean ± standard deviation.
- Thresholds, temperature and other hyperparameters are chosen on the
  validation set only; the test set is used once for final reporting.
- Per-class AUROC and AUPRC, macro averages, bootstrap 95% confidence intervals.
- Classes with too few positives are merged or excluded, with the rule stated.
- Calibration: expected calibration error, reliability diagrams, temperature scaling.

## Repository structure

```
scripts/    numbered pipeline steps
data/       raw and cached images (git-ignored, not distributed)
results/    metrics, tables and figures
notes/      experiment log and decisions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Point the scripts to the dataset location with their command-line arguments.
Heavy steps (image caching and embedding extraction) benefit from a GPU.

## Pipeline

| Step                                    | Purpose                                     | Status  |
| --------------------------------------- | ------------------------------------------- | ------- |
| 1. Metadata analysis                    | Columns, label prevalence, identifiers      | planned |
| 2. Image caching                        | Resize and cache radiographs                | planned |
| 3. Embedding extraction                 | Frozen features per backbone                | planned |
| 4. Heads and metrics                    | Per-class metrics with confidence intervals | planned |
| 5. Calibration and selective prediction | ECE, risk-coverage                          | planned |
| 6. Quality analysis                     | Performance by quality tag                  | planned |

## Related work

- Banzato et al., _Automatic classification of canine thoracic radiographs using deep learning_, Scientific Reports, 2021. doi:10.1038/s41598-021-83515-3
- Burti et al., _Use of deep learning to detect cardiomegaly on thoracic radiographs in dogs_, The Veterinary Journal, 2020.
- Banzato et al., _An AI-based algorithm for the automatic evaluation of image quality in canine thoracic radiographs_, Scientific Reports, 2023. doi:10.1038/s41598-023-44089-4
- Banzato et al., _An AI-based algorithm for the automatic classification of thoracic radiographs in cats_, Frontiers in Veterinary Science, 2021. doi:10.3389/fvets.2021.731936

## Future work

Backbones pretrained on human chest radiographs are not covered here and
are left for follow-up work.

## License

Code license: to be decided. The dataset is distributed under its own terms on Zenodo.
