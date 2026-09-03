# RESISC45: the union head is at a plateau that no knob moves

*2026-09-03. Companion to "Nothing else moves the union head" in RESULTS.md.*

## The question

The union-of-supports head cleared 80% at 3,628 deployed values, but every
knob around it -- the quota list, the source of the diversity, the pruning and
refit `C` -- had been read against single searches, which the same iteration
showed are chaotic. `resisc45_union_variants.py` re-reads them under the
union head itself, with two disjoint search sets per setting, at about 30
seconds a head.

## Findings

**1. The 640-column list is confirmed, readably.** Every wider list
(512/128/128, 384/128/256, 384/256/128, 512/192/192) loses 0.5-2 points under
the union head, and the narrower 256/128/128 list loses 0.5. The single-search
comparison said the same thing but could not be trusted; this one can.

**2. The diversity source only matters through the union's size.** Searches on
50-85% training subsamples, searches over mixed drop/decay settings, and a
two-level union of 32 searches all read 0.7946-0.8010, inside the incumbent's
0.8011 +/- 0.0021. Subsampled and mixed searches disagree more, so their
unions are 17-21k entries against 13k, the pruning discards more than it
keeps, and the mean slips by a few tenths. The two-level union prunes each
group first, lands at 8.9k, and reads the incumbent's mean exactly. The 0.2%
decay perturbation -- the smallest diversity that gives different supports --
is the best of them.

**3. The 80.49% reading has a `C` draw in it.** Every union row picks
`C = 0.03`, the bottom of the refit grid. Extending the grid to 0.003 makes
validation pick 0.01 on the incumbent's own support (0.8195 against 0.8179),
where it reads 0.7995 test. The method is 80.0-80.1% +/- 0.2 at 3,628 values
however `C` is set; 80.49% is the top of that distribution, and the honest
frontier statement in RESULTS.md now says so.

## What this means

On this list at this budget the head is at a plateau: 80.0% against the
list's 81.0% linear ceiling, with an eighth of the dense head's values. The
search is no longer the lever -- the union already averages over its chaos,
and no other way of searching the same 640 columns reads differently. What
would move it is a change to what is being searched: a head with a few learned
hidden units (the list's MLP ceiling is 83%), or new columns that lift the
list's own linear ceiling, found by looking at which classes the 80% head
still confuses.
