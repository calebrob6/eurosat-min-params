# Backbone results

`scoreboard.csv` records frozen-backbone accuracies, and `compute_cost.csv` records their parameter profiles. The archive includes an EarthLoc row omitted from the article's displayed ten-backbone table. These small tables are kept for reference. The public runner defaults to the five backbones used in the feature-combination experiment; `--all-models` includes all eleven archived scoreboard backbones.

The compute profiles use synthetic 12-channel 224x224 inputs. EuroSAT extraction uses 13 channels at 224x224, except OlmoEarth, which omits B10 and uses native 64x64 inputs. Consequently the profiles are not measurements of the extraction workload, and adapted-convolution parameter counts can differ.

`training_fractions/{random,spatial}/` retains the fourteen original sweep tables, restored unchanged from `924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c` (`aed37c9^`). Figures 7/8 use their 94 `method=linear` rows plus four rounded random-split 100% non-Olmo scoreboard values. KNN rows are historical records, not plotted results. The runner writes new curves to ignored `output/backbones/`; it does not overwrite these references.
