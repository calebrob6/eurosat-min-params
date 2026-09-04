# Results used in the EuroSAT blog post

This directory collects the `torchgeo-bench` results used in the geospatialml.com post "Solving EuroSAT with as Few Parameters as Possible."

## Contents

- `scoreboard.csv` contains every frozen-backbone linear-probe row reported in the post's full-data comparison table. Accuracy is stored at the four-decimal precision reported in the post.
- `compute_cost.csv` contains the corresponding Sentinel-2 parameter profiles, filtered from [`torchgeo/torchgeo-bench`](https://github.com/torchgeo/torchgeo-bench) at commit [`9c8e4af`](https://github.com/torchgeo/torchgeo-bench/commit/9c8e4afab46675d7279c88828dfcbf0ca99b3a07).
- `training_fractions/random/` contains the raw CSVs used for the random-split training-fraction figure.
- `training_fractions/spatial/` contains the raw CSVs used for the longitude-based spatial-split training-fraction figure.

## Evaluation protocol

The accuracy runs use the plain 27,000-image EuroSAT dataset rather than the smaller `m-eurosat` GEO-Bench variant. The random split uses 16,200 training, 5,400 validation, and 5,400 test images from the fixed TorchGeo lists. The spatial results use the corresponding `eurosat-spatial` longitude-based partition. All runs set `eval.merge_val=false`, use seed 0, and report 95% confidence intervals from 200 bootstrap resamples.

The random-split ConvNeXt-Tiny, ResNet-50, DOFA Base, and DOFA Large sweep files contain the 1%, 2%, 5%, 10%, 20%, and 50% training subsets. Their 100% endpoints come from `scoreboard.csv`. The three OlmoEarth v1.2 random-split files and all spatial-split files include the 100% run.

The standalone raw full-data outputs for EarthLoc, MoCo ResNet-50, ResNet-18, ResNet-50, ViT Base, ConvNeXt-Tiny, and DOFA Base were not present when this bundle was assembled. Their measured scores are preserved in `scoreboard.csv` at the precision used in the post; missing confidence intervals and optimization metadata have not been reconstructed. The original DOFA Large full-data output remains at `../dofa_large.csv`, and the OlmoEarth v1.2 full-data rows are present in their random-split sweep files.
