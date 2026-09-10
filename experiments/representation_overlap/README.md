# EuroSAT representation overlap

This experiment separates three questions: how much backbone variance handcrafted features predict, which handcrafted measurements are decodable from a backbone, and whether adding handcrafted features improves a controlled linear EuroSAT classifier. It does not estimate mutual information or prove that a backbone lacks information.

The main comparisons use the frozen 33-feature frontier and historical 377-feature pool against ResNet-50, ConvNeXt-Tiny, DOFA Large, and OlmoEarth v1.2 Nano/Base. ImageStats-52 is a secondary baseline. All fits use the official random training split; selection uses validation only. The 33 features are not strictly a subset of the 377: the final region-shape measurement is outside that pool.

## Findings from the locked v2 run

**The engineered features provide substantial practical complementarity, not just a smaller substitute.** Seven of the ten primary accuracy comparisons improve after Holm correction. Adding H33 to ResNet-50 raises accuracy from 95.07% to 98.02%; adding H377 to ConvNeXt-Tiny raises it from 95.15% to 98.17%. H377 also improves DOFA Large by 0.57 percentage points and OlmoEarth Nano by 0.94 points. Neither feature set establishes a reliable accuracy improvement over OlmoEarth Base; the H33 gain for DOFA is also inconclusive. The standalone H33/H377 probes reach 96.04%/97.15%, so the strongest combined models improve on both standalone representations, not merely the weaker backbone.

**Linear overlap is substantial but directional.** H33 explains 25.0%-78.8% of native embedding variance across the five backbones. For example, H33/H377 explain 78.8%/88.7% of OlmoEarth Nano and 59.5%/73.2% of OlmoEarth Base. In the reverse direction, mean per-feature H33 R-squared ranges from 0.75 to 0.90. The full per-model table is in [the generated results summary](results/README.md), with intervals in [decoding.csv](results/decoding.csv).

**Some geometric measurements remain poorly linearly decodable.** Across these backbones, the largest above-median NDVI connected-component fraction and low-NDVI weighted-region anisotropy have per-feature R-squared values of approximately 0.21-0.45. This identifies poorly decoded measurements, not the individual causes of the fusion gains. Class-conditioned decoding also finds shared variation beyond class means: for OlmoEarth Nano, H33/H377 explain 47.8%/69.5% of the class-centered target variance. These percentages use a different denominator from pooled R-squared and must not be subtracted to partition information.

**The controls support a specific, limited interpretation.** Shuffled handcrafted rows and redundant linear projections of the backbone do not reproduce the substantial gains. Removing real B10 information preserves the Nano gains: +0.87/+0.89 points for H33/H377, compared with +0.89/+0.94 with original inputs. These observations support incremental predictive value for the declared linear-probe procedure, not information-theoretic absence from the backbone.

**The ResNet-50 H377 reconstruction is unstable.** Its native R-squared is 23.6%, with a wide 95% bootstrap interval of -1.1% to 36.7%, despite a 34.3% validation score. A post-hoc inspection of the frozen predictions found that `Industrial_2434.tif` alone contributes 15.0% of the total reconstruction squared error; its predicted embedding norm is approximately 123.2 versus an actual norm of 29.3. Several handcrafted coordinates are extreme relative to their training standard deviations. No images were removed, features clipped, or regularization retuned after this observation. This is a concrete warning against interpreting squared-error explanation percentages as robust fractions of information.

All primary, null, band-control, class-conditioned, and named-feature results are retained under [results/](results/). The historical article measurements and checkpoints are unchanged.

## Run

Use the existing pinned `output/benchmark-env` described in `../torchgeo_bench_eurosat/README.md`, from the repository root. No new pretrained extraction is needed when the authenticated `output/benchmark-embeddings/` files are present. Large caches, fitted maps, and sample predictions remain under ignored `output/`.

```bash
# Stream original TIFFs, including a separately named B10-zeroed input ablation.
output/benchmark-env/bin/python -m experiments.representation_overlap.run prepare

# A validation-only pipeline pilot; it cannot be locked or evaluated on test.
output/benchmark-env/bin/python -m experiments.representation_overlap.run select \
  --models olmoearth_nano --pilot --output output/representation-overlap/pilot

# Full selection, without test metrics.
output/benchmark-env/bin/python -m experiments.representation_overlap.run select

# Freeze every selected configuration and artifact before test access.
output/benchmark-env/bin/python -m experiments.representation_overlap.run lock
output/benchmark-env/bin/python -m experiments.representation_overlap.run evaluate

# Export a fresh report without replacing the checked-in study snapshot.
output/benchmark-env/bin/python -m experiments.representation_overlap.run report \
  --export output/representation-overlap/report
```

`select --component decoding` and `select --component probes` can be run separately; both must finish before locking. `--output` identifies a study run, and changed code/configuration/input hashes require a new directory. Completed selection jobs can resume only under the same protocol. Export destinations also reject a different run rather than overwrite it; the command above keeps fresh reports in ignored output. A mismatched backbone checksum is an error, not a reason to silently overwrite embeddings. Alternative embedding files and their corresponding JSON metadata must be supplied together with `--embeddings` and `--metadata`.

## Measurement definitions

**H -> Z:** an affine ridge map predicts native backbone coordinates from train-standardized handcrafted inputs. The main score is `1 - total squared prediction error / total centered target sum of squares` on the held-out split. Report its percentage as *embedding variance linearly explained*, not percent of information, coordinates, or backbone parameters. Per-coordinate scores expose variation hidden by the native variance-weighted aggregate.

**Z -> H:** predict each named handcrafted measurement from train-standardized backbone coordinates. Use macro per-feature validation R-squared to select regularization, so reflectance units cannot dominate dimensionless texture measurements. Negative scores remain negative. Constant targets have undefined per-feature scores and explicit counts, not forced zeros or ones.

These decoders are fitted analysis models, not zero-parameter feature extractors or the published 306-value classifier. Predicting an embedding with a learned map does not establish a parameter-free replacement for the backbone. Nor is every comparison dimensionality reduction: H377 has more coordinates than the 128-dimensional Nano embedding.

**Incremental classification:** compare the same backbone-only linear-probe procedure with block-controlled concatenation. The backbone keeps its relative coordinate scaling; handcrafted coordinates are training-standardized. Each block is normalized by its total training variance. Select C and the handcrafted block weight on validation accuracy, with log loss secondary. The grid includes zero handcrafted weight. A fixed linear projection of the backbone and shuffled handcrafted rows are additional controls for regularization and extra-feature effects.

The estimator is the pinned `torchgeo_bench.linear.LogisticRegression`, not a new optimizer. It uses train only, with explicit device selection and TF32 disabled. The selection and test stages are separate rather than calling the benchmark's combined `evaluate_logistic`.

The v2 protocol gives every fit 8,000 iterations and restarts once with 16,000 if either the iteration or function-evaluation budget is exhausted, then fails explicitly. Tolerances remain unchanged. Returning before those budgets is recorded as such, not as proof that a gradient threshold was reached. The initial v1 run was stopped before test evaluation when one high-C shuffled control exceeded its 2,000/4,000-iteration budgets; those artifacts are preserved separately rather than relabeled.

Grid-boundary selections are explicit outcomes, not claims of globally optimal regularization. The secondary ImageStats decoders can approach the unregularized limit, and null decoders can prefer the strongest regularization. The classification conclusions apply to the declared C/block-weight grid.

**No residual novelty test:** for a fixed affine decoder, `[Z, H - ZA - b]` and `[Z, H]` support the same affine score functions. Residual concatenation is therefore not independent evidence that information is missing. Ridge residuals are not generally orthogonal to Z either.

## Diagnostics and limitations

The class-conditioned decoder subtracts training class means from both representations. It intentionally uses known class labels at evaluation and is an oracle-label diagnostic, not a deployable feature transformation. It asks whether overlap extends beyond average differences between EuroSAT classes. Row-permutation controls expose finite-sample and class-membership effects.

Train-fitted PCA gives a variance spectrum and rank-matched target-reconstruction reference. The reference sees the actual target embedding at evaluation; it is neither an H-to-Z predictor nor a guaranteed held-out ceiling. Prediction quality need not improve monotonically with rank. Linear CKA is computed without a sample-by-sample Gram matrix, using standardized H and native Z; it measures symmetric geometry rather than containment.

OlmoEarth omits B10. The `b10_zeroed` control replaces physical B10 at TIFF index 9 with a fixed zero before **all** handcrafted arithmetic, including the panchromatic mean. It keeps 13 channel positions and does not redefine the frozen recipes. It is a separately fitted input ablation, not the published checkpoint. Historical SWIR/index aliases retain their numeric meanings; exported feature tables include physical-band definitions.

The B10 ablation creates constant targets in H377, so its macro reverse-decoding average covers fewer defined measurements. A higher average is not itself evidence of better representation quality; compare corresponding nonconstant features and the controlled classification gains.

The ten primary accuracy comparisons are two feature sets times five backbones. Paired bootstrap intervals use image IDs, not embedding dimensions; exact McNemar tests receive Holm adjustment across the primary comparisons. ImageStats, sham additions, band ablations, geometry, and per-class/family results are secondary. Intervals are conditional on fitted models and do not cover training-set variation, feature-discovery bias, or spatial dependence.

EuroSAT and these handcrafted families were extensively studied before this follow-up. The protocol prevents new test-guided choices but does not turn the dataset into an untouched external benchmark. A gain demonstrates incremental predictive value for this linear-probe procedure; no gain does not prove complete redundancy, and poor linear decoding does not rule out nonlinear encoding.

## Outputs

The exported results contain all validation candidate scores in `selection.csv`, full decoding and classifier tables, paired accuracy/log-loss differences, B10 controls, per-class counts, named-feature/family decodability, CKA controls, PCA references, paired H377-minus-H33 reconstruction differences, and figures. Undefined numeric entries are empty in CSV and null in JSON. Source, input, feature-schema, package, and protocol hashes are recorded separately from historical article results.

The current grids and deterministic settings live in `protocol.py`. Selection logs record every candidate and the numerical solver's iteration status; null or negative results are retained rather than omitted. The report exports all locked conditions or fails on incomplete coverage.
