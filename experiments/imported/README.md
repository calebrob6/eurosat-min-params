# Original EuroSAT benchmark outputs

These files were supplied from the original TorchGeo-bench experiment checkout. They are historical evidence, not the fresh rerun. CSVs and manifests are preserved without changing their measurements; local logs, lock files, and bytecode are excluded from Git.

| Directory | Contents used by the article |
|---|---|
| `eurosat-13band-merge-val-false-20260901/` | Eight full-data backbone comparisons |
| `eurosat-train-fractions-20260901/` | ResNet-50, ConvNeXt-Tiny, DOFA Base/Large random-split curves, 1-50%, plus the original sampling script |
| `eurosat-spatial-train-fractions-20260901/` | The same four models, spatial curves, 1-100% |
| `olmoearth-train-fractions-20260901/` | OlmoEarth version/size comparisons; the article uses v1.2 Nano, Small, and Base |

The original `run_fraction_sweep.py` extracts the shuffled training loader once, then uses prefixes of `np.random.default_rng(0).permutation(n_train)`. These are **nested random subsets, not stratified subsets**. The script snapshot predates the `--dataset`, `--fractions`, and `--skip-knn` arguments used by the accompanying OlmoEarth shell script. For a supported runner covering both protocols and all seven fractions, use [`../torchgeo_bench_eurosat/run.py`](../torchgeo_bench_eurosat/run.py).

Fresh measurements and original-vs-fresh differences are in [`../torchgeo_bench_eurosat/reproduced/`](../torchgeo_bench_eurosat/reproduced/). The untouched imported Python script was also run with ResNet-50 as a control: all six linear accuracies and chosen C values matched the new runner exactly in the current environment.
