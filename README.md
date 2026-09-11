# Joining two separate measurements into one 2-D picture

## What this does

We measured two different things: **intensity** 36 times, and a **z score** 10 times.
This script finds the best curve for each one separately, then glues the two curves
together into a single 2-D picture.

The measurements are invented by the script itself, so you can just run it.

---

## Run it

```bash
python3 copula_joint_fit.py
```

Takes about a minute. Almost all of that is step 5 (the bootstrap), which refits the
curves 400 times over.

It prints a running commentary and writes **`copula_joint_fit.png`** next to itself.

## What the picture shows

| Part of the figure | What it is |
|---|---|
| **Big middle panel** | The 2-D answer. The rings are contours — see the glossary. |
| **Blue circle** | The average: the typical intensity paired with the typical z score. |
| **Top strip** | Intensity's own curve, with its 36 measurements below it. |
| **Right strip** | The z score's own curve, with its 10 measurements beside it. |
| **Grey bars** | A histogram — how many measurements landed in each bucket. |
| **Orange ticks** | Every individual measurement, one tick each. |
| **Pale blue shading** | How much the curve would have wobbled with different luck. Wider = less certain. |

There are deliberately **no scattered dots** in the middle panel. Plotting a dot needs
both an intensity *and* a z score to position it, and we never measured the two
together. More on that below.

---

## Words you need

Everything below is also written at the top of the `.py` file, next to the code.

| Term | In plain words |
|---|---|
| **Probability density** (curve, "pdf") | A curve showing which values are common and which are rare. Tall = common. The area under the whole curve is always exactly 1, because the measurement has to come out as *something*. Height alone means nothing — only **area** is a probability. |
| **Running total** ("CDF") | Sweep left to right piling up area as you go. `CDF(0.3) = 0.45` means 45% of measurements come in below 0.3. Starts at 0, ends at 1. |
| **Sample** | The numbers we actually measured. A blurry snapshot of the real curve; more points means a sharper snapshot. |
| **Fitting** | Picking the curve that best explains the numbers we saw. |
| **Normal** | The ordinary symmetric bell curve. **2 dials:** where it sits, how wide it is. |
| **Skew normal** | A bell curve allowed to lean. **3 dials:** where, how wide, and how hard it leans. Lean of 0 *is* an ordinary normal. |
| **Likelihood** | "If this curve were the truth, how unsurprising would our measurements be?" Higher = better explanation. |
| **Truncation** | Chopping a curve off at a hard physical limit and scaling up what's left so the area is 1 again. |
| **AIC** | The score used to choose between two curves. Lower wins. Explained in full below. |
| **Bootstrap** | Re-running the fit on shuffled versions of the data to see how much the answer wobbles. |
| **Copula** | Glue. Joins two finished 1-D curves into one 2-D surface *without bending either one*. |
| **Contour ring** | The "68% ring" is the smallest region you can draw that still contains 68% of all outcomes. |

---

## What the script does, step by step

### Step 1 — Invent the two sets of measurements

Two bags of numbers get generated from known curves, so you can watch the script try
to recover those curves later.

**Intensity (36 numbers).** Intensity physically cannot leave the range 0 to 1, but a
skew-normal curve stretches to infinity in both directions. So instead of drawing
freely and throwing away whatever misses, the script draws from the *chopped* curve
directly, using the running total:

1. Work out the running total at 0 and at 1 — say those are 0.003 and 0.998.
2. Pick a random number between those two.
3. Run the total backwards to find which intensity gives that number.

Every draw lands inside the bounds first try, with nothing wasted.

**Z score (10 numbers).** No hard wall here, so an ordinary random draw is fine. The
`-3 to 3` range is just the plotting window, not a physical limit.

**They are not paired.** This is the whole reason the script is shaped the way it is.
We have 36 of one thing and 10 of another, and no record anywhere saying "*this*
intensity happened at the same time as *this* z score." So the data literally cannot
tell us how the two relate. Nothing later in the script can conjure that information
out of nothing.

### Step 2 — Fit each measurement on its own

Each bag is fitted completely separately; nothing about intensity touches the z-score
fit or the reverse. Each one runs the same three substeps.

**2a. Fit an ordinary normal.** Search for the 2 dial settings that best explain the
numbers.

**2b. Fit a skew normal.** Same, but with a 3rd dial for the lean. Two details make
this work:

- *Several starting guesses, not one.* A lean of exactly 0 is a perfectly flat spot in
  the search landscape — an optimiser dropped exactly there sees no downhill direction
  and never moves. The script starts from six different leans straddling zero.
- *A truncation correction, for intensity only.* Chopping the curve at 0 and 1 throws
  away area, so the chopped curve is **taller** everywhere inside (its area still has
  to total 1). The score has to account for that. Leaving it out breaks the fit in a
  specific way: the optimiser gets rewarded for shoving the curve's bulk outside the
  bounds where nothing needs explaining, so it drifts outward.

**2c. Pick the winner using AIC.** Lower score wins — plus one extra rule that turns
out to matter enormously:

> If the fitted lean runs off to the safety rail, throw the skew normal out entirely.

Here is why. With very few measurements, the skew-normal search never settles. It
leans harder and harder toward infinity, because a curve leaning infinitely can be
bent through a handful of points. Unchecked, it posts a fantastic score and **wins**
with a curve that is complete nonsense.

Measured on 40 random 10-point samples: the runaway fit won 25 times, and **all 25**
had a lean around 1,000,000,000,000,000. So a lean jammed against the rail means the
data never pinned it down. Discard it rather than believe it.

### Step 3 — Work out the average

The blue circle. Two different methods, depending on the curve:

- **No truncation** (the z score): there is an exact formula, so it just uses it.
- **With truncation** (intensity): no such formula exists, so it uses brute force —
  chop the range into 20,000 slivers, add up (value × height) across all of them, and
  divide by the total height. The dividing cancels out most of the sliver-counting
  error.

The average **does not move when you change `RHO`**. Changing how two things relate
does not change either one's average.

### Step 4 — Glue the two curves together

This is the copula. The formula:

```
f(x, y) = c(F1(x), F2(y)) * f1(x) * f2(y)
```

Term by term:

- `f1(x)`, `f2(y)` — the height of each finished 1-D curve
- `F1(x)`, `F2(y)` — each curve's running total, a number between 0 and 1
- `c(...)` — the **glue factor**, and the only place `RHO` acts

When `RHO` is 0, the glue factor is exactly 1 everywhere and the formula collapses to
plain multiplication — the two measurements are independent. Turn `RHO` up and the
glue grows along one diagonal and shrinks along the other, tilting the surface, while
**leaving both original curves perfectly intact**.

`RHO` is the one dial you set by hand:

| `RHO` | Meaning |
|---|---|
| `0.0` | Unrelated. The 2-D answer is just curve1 × curve2. *(default)* |
| `+0.6` | High intensity tends to come with a high z score. |
| `-0.8` | High intensity tends to come with a low z score. |

**`RHO` cannot be measured from this data** — see the unpaired point in Step 1. It is
your assumption about the system.

The script then prints four self-checks:

1. **Does the surface add up to 1?** It has to; something always happens.
2. **Squash it flat onto the intensity axis** — intensity's original curve must come
   back unchanged.
3. **Squash it onto the z-score axis** — same test the other way.
4. **With `RHO = 0`, is it exactly height × height?** Must be exactly zero difference.

Checks 2 and 3 are the copula's defining promise. If those fail, the glue is wrong.

One subtlety worth knowing: the checks run on a **wider grid** than the one drawn. The
z-score curve isn't truncated, so a sliver of it lives outside the plotted `-3 to 3`.
Adding up only what's on screen would come to slightly under 1 and make a perfectly
correct surface look broken — by exactly the amount left off screen.

### Step 5 — Find out how much each sample size actually buys

This is the slow part, and it produces the pale blue shading.

For each measurement, 200 times over:

1. Build a fake sample the same size as the real one by drawing from the real
   measurements at random **with replacement** — so some get picked twice and others
   not at all, mimicking the luck of which units you happened to test.
2. Refit the curve to that fake sample.
3. Record the result.

Then at every point, report the range that 90% of those 200 refits fell inside. That
is the shaded band.

**A wide band is not a bug.** It is the honest statement that this many points could
not narrow things down any further.

### Step 6 — Draw it

Contour rings for the 2-D surface, and each 1-D curve on the edge it belongs to.

Finding the "68% ring": sort every point on the surface tallest-first, then walk down
that list adding up area. The height you've reached when the total hits 68% is the
ring's height. Working tallest-first is what makes the ring hug the peak instead of
being some arbitrary shape that happens to enclose 68%.

---

## Why the z score comes out as a normal, not a skew normal

Because 10 points cannot tell the difference.

The script fits both and the skew normal's lean runs off to infinity, so it gets
disqualified (Step 2c). That isn't bad luck with this particular dataset: across 40
different random 10-point samples — **all drawn from genuinely lopsided data** — the
skew normal was never the right call. **0 times out of 40.** With 36 points it works
fine, which is why intensity does get one.

The shaded bands show the same thing directly. Relative to its own height, the
z-score band is about **107%** of the peak versus **71%** for intensity. That extra
width is the honest cost of having 10 measurements instead of 36.

## What is AIC?

The score used to choose between the normal and the skew normal. **Lower is better.**

```
AIC = 2*(number of dials) - 2*(how well it fits)
```

A curve with more dials can always hug the data more closely, so "which fits better?"
is a rigged question. AIC charges **2 points per dial**, so extra complexity has to
earn its place. A normal has 2 dials and a skew normal has 3, so the skew normal
starts 2 points behind. A gap under 2 means "can't really tell these apart."

## Why not a multivariate skew-normal?

That was the original plan and it doesn't work here. Two reasons.

**It can't keep both fits.** In a multivariate skew-normal, one parameter controls
*both* how lopsided each variable is *and* how the two relate. You can't set one
without disturbing the other. Concretely, if the two fitted leans are `a1` and `a2`,
an uncorrelated multivariate skew-normal only exists when

```
|a1 * a2| < 1
```

So two clearly-lopsided measurements can never be uncorrelated. Leans of 3.0 and -1.5,
for example, have no solution at all. The copula has no such limit — it works for any
two curves and any `RHO`.

**The data isn't paired.** Even if the math worked, there'd be nothing to fit the
relationship to.

The scripts in the parent folder show what a multivariate skew-normal looks like, kept
purely as a reference. They are not the right tool for this data.

---

## Knobs you can change

All at the top of `copula_joint_fit.py`:

| Setting | Does what |
|---|---|
| `RHO` | How much the two move together. `0` = independent. |
| `N_INTENSITY`, `N_ZSCORE` | How many points to invent. Raise `N_ZSCORE` to about 100 and the z score starts coming out as a skew normal instead (at 50 it usually still doesn't). |
| `TRUE_INTENSITY`, `TRUE_ZSCORE` | The real shape of the invented data (`xi` = centre, `omega` = width, `alpha` = lean). |
| `INTENSITY_BOUNDS` | The hard 0-to-1 limit on intensity. |
| `N_BOOTSTRAP` | Resamples for the shaded band. Lower it to make the script faster. |
| `SEED` | Change for a completely different random dataset. |

## Notes

- The **blue circle** is the average of the fitted 2-D surface. It does not move when
  you change `RHO`. The script also prints the raw measured averages next to it for
  comparison.
- The z score is **not** cut off at ±3 — that's the plot window, not a hard limit.
  Only intensity is truly bounded. If ±3 *is* a real limit for you, pass
  `ZSCORE_RANGE` as `bounds` when the z-score curve is built.
- The self-checks in Step 4 should print errors of roughly `1e-6` or smaller. Anything
  larger means something is wrong.
- Needs `numpy`, `scipy` and `matplotlib`.
