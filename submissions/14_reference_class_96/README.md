# 33-feature EuroSAT model: 306 learned values

This is the frozen reference-class logistic model used in "Solving EuroSAT with as Few Parameters as Possible": 33 fixed measurements, nine stored class rows, and **9 x (33 + 1) = 306** learned weights and biases.

| Split | Correct / images | Accuracy |
|---|---:|---:|
| Validation | 5,193 / 5,400 | 96.17% |
| Test | 5,186 / 5,400 | 96.04% |

The checkpoint stores the selected feature order, extraction configuration, physical TIFF band order, historical channel aliases, and folded weights. `src/frontier.py` recreates the feature subset without the old mega-pool caches. **Do not substitute different SWIR indices or "correct" the historical spectral-index aliases:** see [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) for the physical-band interpretation.

From the repository root:

```bash
python reproduce.py --download
```

This evaluates the checked-in head from the raw TIFFs, along with the 171- and 279-value heads and ImageStats. For a train-only refit with the fixed feature subset and C=3:

```bash
python reproduce.py --download --refit-306 --output output/refit
```

Refitting writes `output/refit/model_306.npz`; it never overwrites this checkpoint. It does not rerun the adaptive historical feature search. The experiment traces under `experiments/*ceiling96*` record how the subset was chosen.

For only this saved model, without any fitting:

```bash
python submissions/14_reference_class_96/eval.py
```
