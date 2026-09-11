# Backbone results

`scoreboard.csv` records frozen-backbone accuracies, and `compute_cost.csv` records their parameter profiles. The archive includes an EarthLoc row omitted from the article's table.

`training_fractions/{random,spatial}/` holds the training-fraction sweeps behind the article's learning-curve figures. Those figures use the `method=linear` rows, plus the full-data scoreboard values for the four non-OlmoEarth random-split 100% points.

The runner writes new measurements to `output/backbones/`; it does not overwrite these tables.
