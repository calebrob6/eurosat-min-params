# Corner/junction family pre-screens stronger than the line family did — descending

*2026-07-02 01:05 — iteration 15 (in progress)*

## Salvage: what iteration 14 left behind

Iteration 14 crashed (no notes entry, sources reverted) but its *caches*
survived in `data/cache/`: a 24-feature `gstruct` block (4 candidate global-
structure families × pan/NDVI/NDBI) and a `blobfam_blob` extraction, plus the
`.pyc` of a half-built `submissions/11_blob_backward_linear` whose constants
show it was committing to the **blob** family (o6+line+blob pool, `--k 20`).
The generating source for those caches is lost — so nothing built on them
would be reproducible, and they can only be used for *screening*.

## Pre-screen (the iteration-13 qualifier) on all 24 lost-cache candidates

Add ONE candidate feature to the FROZEN submission-10 subset (k=23, val
0.9404), read held-out val: `experiments/gstruct_prescreen.py`
(`GS=old`, results in `gstruct_prescreen_result.txt`):

| family | best member | dval | family consistency |
|--------|-------------|------|--------------------|
| corner | cornfrac_ndvi | **+0.0044** | all 6 lift val (+0.0006..+0.0044) |
| lbp | lbpuni_ndvi | +0.0039 | all 6 lift val (+0.0024..+0.0039) |
| blob | bloblrg_ndvi | +0.0030 | mixed (3 of 9 ≥ +0.0019) |
| specslope | specslope_ndvi | +0.0019 | weak |

**Iteration 14 bet on the wrong family.** Blob (its choice) is third; corner
matches the line family's winning pre-screen (+0.0042) and lbp is close.
Family-wide consistency matters: 24 candidates invite a multiple-comparisons
false positive, but a real orthogonal axis lifts val across all its members
and channels, which corner and lbp do and blob does not.

## Reimplementation (`experiments/gstruct2_features_lib.py`)

Clean-room reimplementations of all 4 concepts (blob2/corn2/lbp2/sslope2),
cached under `gs2fam_*`. Value ranges land almost exactly on the lost cache
(lbp entropy max 2.236 vs 2.2366 — the canonical riu2-10-bin definition;
blob lrg capped at 0.5 = above-median mask). Re-screen with MY features
(`gstruct_prescreen_gs2_result.txt`) is even stronger for corner:

- `corn2mag_ndbi` **+0.0046 val**, `corn2frac_ndbi` +0.0044,
  `corn2frac_ndvi` +0.0041 with **+0.0039 cv5** (the largest CV lift of any
  candidate this project has screened; line's was +0.0017).
- All 6 corn2 features lift val. lbp2: all 6 lift (+0.0022..+0.0043).
  blob2: only `blob2lrg_ndvi` strong. sslope2: fails (≤ +0.0009).

Corner mechanism: corners are where edges MEET — a global-layout axis nothing
in the pool sees. Local directional stats aggregate isolated gradient
directions; Hough sees straight lines; neither separates a junction grid
(Residential/Industrial blocks) from parallel rows that never cross
(AnnualCrop). Promoted bit-identically into
`src/features.py::harris_corner_features` (`patch_features(harris_corners=True)`,
pool order o6[0:305]+line[305:314]+corn2[314:320]).

## In flight

Two-partition FORCE forced-union backward descents on `o6+line+corn2`
(dim 320, |init|=54, START=42, STOP=19, C=10, seeds 0-9/10-19 and 20-29/30-39),
results streaming to `backward_select_o6_line_corn2_p{0,20}force_result.txt`.
New `DIAG_MIN_K=30` env on `backward_select.py` skips the per-row
verCV/val/test diagnostics above k=30 (they never steer the path) — saves
~1-2 min/step on this loaded machine (load avg ~104/40 cores).

If the honest floor (verCV≥0.94 AND val≥0.94 on BOTH partitions) lands at
k≤22, submission 11 goes below 240 params. lbp2 is the pre-qualified backup
family if corner disappoints; blob2/sslope2 are screened out as solo families.

## Cross-family stacking probe (the iteration-16 lever)

Two-feature additions to the frozen-23 subset (val fits, C=10):

| addition | k | val |
|---|--:|--:|
| (frozen baseline) | 23 | 0.9404 |
| corn2frac_ndvi | 24 | 0.9444 |
| corn2frac_ndvi + corn2mag_ndbi (within-family) | 25 | 0.9461 |
| corn2frac_ndvi + lbp2uni_ndvi | 25 | 0.9480 |
| **corn2frac_ndvi + blob2lrg_ndvi** | 25 | **0.9493** |
| corn2frac_ndvi + cm_ndbi + lu_ndvi | 26 | 0.9489 |

Cross-family stacking is nearly ADDITIVE (+0.0089 val from two features) while
within-family stacking saturates — the global-structure axes (junctions,
micro-pattern, region granularity) are mutually orthogonal, unlike iteration
5's weak families. blob2lrg_ndvi is weak SOLO but its residual against corner
is strong. If the corn2-only descent lands k≤22 this iteration, the natural
next lever is one descent with ALL of corn2+lbp2+blob2lrg forced
(o6+line+corn2+lbp2+blob2, |init|≈66) aiming at k≈20 (210 params) — the
+0.009 val headroom is ~9× the ~0.001 the val gate needs per step.
