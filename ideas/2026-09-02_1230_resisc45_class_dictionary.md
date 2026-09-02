# One stored value, many classes

*2026-09-02 12:30 — RESISC45 iteration 6*

## The idea

Iteration 5 closed with a diagnosis rather than a result: above about 512 stored
values the RESISC45 head is *weight-limited, not column-limited*. It already
reads 348 distinct columns at 512 values and 635 at 1,024, far more than any
class can afford to combine, so a better pool, a better column, or a wider
candidate list cannot relieve the binding constraint. Closing the remaining
7-point gap to the pool ceiling needed "something that changes how many weights
a class effectively has".

The element-wise sparse head spends one stored value on one `(class, column)`
weight. A column that eight classes want therefore costs eight values, and with
45 classes and 1,024 values each class can afford about twenty-two weights. So:
generate the head instead. Write `W = D @ P` with `D [44, atoms]` a **fixed**
dictionary of unit-norm class directions and `P` sparse. A stored value now buys
a whole pattern across classes. If `D` is generated from a seed by a stated rule
it is as free as the feature extractors — deterministic arithmetic, no learned
constants — and the accounting is unchanged: `nnz(P) + 44` biases.

The 44 identity atoms are a valid dictionary, and that arm reproduces the
incumbent head bit for bit — same support, same accuracy — which makes every
comparison a controlled one.

## What happened

It is the largest single RESISC45 improvement in the run. The smallest budget
that clears 65% test falls from 512 to **256 stored values**, the smallest that
clears 70% falls from 896 to **448**, and the 1,024-value operating point goes
from 72.90% to **75.76%** — a paired-bootstrap gain of +9.98 points [+8.9,
+11.2] at 256 values, +6.4 at 512 and +2.9 at 1,024, with three independent
dictionary draws within 0.8 points of each other at every budget.

## Which part of the idea was load-bearing

Not the part I expected. The motivating story was *semantic sharing*: water
classes should be able to buy a greenness weight together. Three controls say
otherwise.

* **A complete basis is worth nothing.** Rotating class space into the dense
  head's own principal directions, or into a DCT basis, moves test accuracy by
  -1.0 to +0.9 points with no consistent sign. That is the right answer: a
  complete basis does not change the model class, it only relabels which
  patterns are cheap, and apparently no 44-dimensional labelling is much better
  than the identity.
* **Which groups almost does not matter.** Ward class groups from the pool beat
  size-matched *random* groups by 0.9 points at 256 values and by 0.03 at 1,024,
  and groups derived from the dense head's own rows — the structure the code is
  literally approximating — are the worst of the three.
* **How many atoms matters a lot.** 44 → 1,024 → 4,096 → 16,384 atoms buys
  +6.2, +2.7 and +1.0 test points at 256 stored values, flattening by about
  16,384, where the search picks 980 entries out of 8.4 million.

So the mechanism is not semantics but **over-completeness**: with far more atoms
than classes, the search can *choose* a class pattern that fits each column's
job instead of paying per class for a pattern it is forced to spell out. It is
compressed sensing on the head — the deployed matrix is dense (17,600 nonzeros
at 1,024 stored values) and lives in an adaptively chosen 980-dimensional slice
of the 22,572-value dense head.

That also explains, retrospectively, why iteration 2's block-sparse head and
iteration 5's bagged supports both failed. Both tried to buy sharing by
*constraining* classes to agree; this buys it by *widening* the alphabet the
code may spell a class pattern with, and letting the conditional regrow
criterion pick.

## What it costs, honestly

Nothing in stored values and nothing in extraction breadth (400 pool columns
against 383 at 1,024 values), but two things do grow. Inference becomes a dense
`45 x 512` product rather than a sparse gather. And each stored value needs a
wider atom id — `log2(16384)` bits rather than `log2(44)` — taking the index
pattern from 14.2 to 22.5 kbit at 1,024 values. Read at **equal total bits**
instead of equal stored values the head is still ahead by +9.4 points at 0.74
KiB and +3.4 at 5.0 KiB, so it pays for its own indices with room to spare, but
that is the comparison a sceptic should be shown first, and `--summarise` prints
it.

## What this suggests next

* The candidate list is doing more work again. For the dictionary head the
  512-column list beats the whole pool by 1.1-2.0 test points at every budget
  from 640 upwards, where iteration 4 found the two within 1.5 points and the
  whole pool winning outright below 512. The code now searches `atoms x
  columns`, so a wider column list multiplies an already huge space. A ranking
  that is *conditional* rather than label-only would be worth more here than it
  ever was.
* The per-class picture has not been looked at since iteration 2. The budget
  used to starve bridge, ship and railway_station specifically; with three times
  the effective weights those should be the classes that moved, and if they did
  not, the object-layout pool deserves a second look.
* Nothing here is RESISC45-specific. EuroSAT has 10 classes, so the head is a
  far smaller fraction of its budget and the same trick should be worth much
  less — but "much less" at 171 parameters is still worth measuring, and it is a
  cheap experiment.
