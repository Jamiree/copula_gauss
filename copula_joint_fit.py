"""
================================================================================
 JOINING TWO SEPARATE MEASUREMENTS INTO ONE 2-D PICTURE
================================================================================

THE SITUATION
-------------
We measured two different things:

    intensity  -- 36 times, always a number between 0 and 1
    z score    -- 10 times, typically between -3 and +3

Crucially, these are NOT paired. We do not have 36 rows each holding an
(intensity, z score) couple. We have one bag of 36 numbers and a separate bag
of 10 numbers, and nothing links a specific intensity to a specific z score.

WHAT THIS SCRIPT DOES
---------------------
    1. Invents the two bags of numbers (so you can just run this file).
    2. Finds the best curve for each bag, one bag at a time.
    3. Glues the two curves into a single 2-D surface using a "copula".
    4. Measures how trustworthy each curve is, given how few points it had.
    5. Draws the result.

Read README.md alongside this file. It says the same things with pictures and
without code.

================================================================================
 GLOSSARY -- every idea used below, defined once, in plain words
================================================================================

PROBABILITY DENSITY (also "density", "curve", "pdf")
    A curve showing which values are common and which are rare. Where the curve
    is tall, that value shows up often; where it is low, that value is rare.
    The area underneath the whole curve is always exactly 1, because the
    measurement has to come out as *something*.

    A density is NOT a probability. Height alone means nothing; only AREA is a
    probability. The area under the curve between 0.2 and 0.3 is the chance of
    landing between 0.2 and 0.3.

RUNNING TOTAL (the "CDF")
    Start at the far left and sweep right, accumulating area as you go. That
    accumulated area is the CDF. It starts at 0, ends at 1, and never goes
    down. CDF(0.3) = 0.45 means "45% of measurements come in below 0.3".

    Its reverse ("ppf" in the code) answers the flipped question: what value do
    45% of measurements fall below? Handy for generating fake data.

SAMPLE
    The actual numbers we measured. Our two samples are 36 numbers and 10
    numbers. A sample is a blurry snapshot of the underlying curve; more points
    means a sharper snapshot.

FITTING
    Choosing the curve that best explains the numbers we actually saw. You pick
    a *family* of curves (see below), then search inside that family for the
    single best member.

THE NORMAL -- 2 dials
    The ordinary symmetric bell curve.
        location = where its middle sits
        scale    = how wide it is
    Symmetric: the left half mirrors the right half.

THE SKEW NORMAL -- 3 dials
    A bell curve allowed to lean.
        location = roughly where it sits
        scale    = roughly how wide
        shape    = how hard it leans. 0 = no lean, so a skew normal with
                   shape 0 IS an ordinary normal. Positive leans right (a long
                   tail stretching to high values), negative leans left.
    Because shape=0 recovers the normal exactly, the normal is a special case
    of the skew normal. That matters for the AIC comparison below.

LIKELIHOOD, and LOG-LIKELIHOOD
    "If this exact curve were the truth, how unsurprising would our measured
    numbers be?" Higher = the curve explains the data better.

    To score a whole sample you multiply one number per measurement together.
    Multiplying 36 small numbers underflows to zero on a computer, so we add
    their logarithms instead. Same ranking, no underflow. That sum is the
    log-likelihood, and every fit below is a hunt for the dials that make it
    as large as possible.

OPTIMISER
    The search itself. We hand it a starting guess and it walks downhill until
    it cannot improve. It minimises, so we feed it NEGATIVE log-likelihood --
    minimising the negative is the same as maximising the original.

    An optimiser can get stuck in a dip that is not the deepest dip. Defence:
    start it from several different places and keep the best finish.

TRUNCATION
    Intensity physically cannot leave [0, 1], but a skew-normal curve stretches
    to infinity in both directions. So we chop the curve off at 0 and 1 and
    scale up whatever survives, restoring its area to exactly 1. That scaling
    up is "renormalising".

AIC (Akaike Information Criterion)
    A score for comparing two fitted curves. LOWER IS BETTER.

        AIC = 2*(number of dials) - 2*(log-likelihood)

    A curve with more dials can always hug the data more closely, so raw fit
    quality is a rigged contest. AIC charges 2 points per dial, forcing extra
    complexity to earn its place. Normal = 2 dials, skew normal = 3, so the
    skew normal starts 2 points behind. Gap under 2 means "cannot really tell
    these apart".

BOOTSTRAP
    A way to ask "how much would this fit have wobbled if we had been unlucky?"
    Draw a new fake sample the same size as the real one by picking from the
    real numbers at random WITH replacement (so some appear twice, others not
    at all). Refit. Do it 200 times. The spread of those 200 curves is the
    shaded band on the plot. A wide band means the data barely pinned the curve
    down.

COPULA
    Glue. It takes two finished 1-D curves and builds one 2-D surface from
    them, WITHOUT bending either curve out of shape. Squash the finished
    surface back down onto either axis and you get that axis's original curve
    back, untouched.

    That separation is exactly what we need: each measurement's shape is
    settled by its own data, and how the two relate is a separate dial (RHO)
    that we set by hand. See README.md for why a multivariate skew-normal
    cannot do this -- in that model, one parameter controls both at once.

HIGHEST-DENSITY CONTOURS
    The rings on the plot. The "68%" ring is drawn so that the region inside it
    holds 68% of the total area, and so that every point inside is taller than
    every point outside. It is the smallest region you can draw that still
    captures 68% of the outcomes.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm

# Save output next to this script, wherever the folder gets moved to. Using a
# hardcoded path here would silently write the PNG somewhere else after a move.
OUT_DIR = Path(__file__).resolve().parent

# ============================================================================
# KNOBS -- the things you are meant to change
# ============================================================================

# How strongly intensity and z score move together, from -1 to +1.
#     0.0  = unrelated. The 2-D answer is literally curve1 x curve2.
#    +0.6  = high intensity tends to come with high z score.
#    -0.8  = high intensity tends to come with low z score.
#
# IMPORTANT: this is NOT measured from the data and cannot be. Our two samples
# are unpaired, so they hold no information whatsoever about how the two
# quantities relate. RHO is your assumption about the system. Default 0.
RHO = 0.0

N_INTENSITY = 36    # how many intensity measurements to invent
N_ZSCORE = 10       # how many z-score measurements to invent

# The "true" curves the fake data is drawn from. The script then tries to
# recover these by fitting, so you can see how close it gets.
#   xi    = location (where it sits)
#   omega = scale    (how wide)
#   alpha = shape    (how hard it leans; + leans right)
TRUE_INTENSITY = dict(xi=0.15, omega=0.28, alpha=3.5)
TRUE_ZSCORE = dict(xi=-0.5, omega=1.2, alpha=1.5)

# Intensity genuinely cannot escape [0, 1], so its curve is truncated there.
INTENSITY_BOUNDS = (0.0, 1.0)

# For the z score, +/-3 is just the plotting window, NOT a physical wall, so its
# curve is left untruncated and is allowed a little tail past the edges.
ZSCORE_RANGE = (-3.0, 3.0)

# Safety rail. If a fitted "shape" dial ends up bigger than this, it means the
# search ran away to infinity, which is what happens when there is not enough
# data to pin the lean down. We treat that as "no answer" rather than an answer.
ALPHA_BOUND = 20.0

N_BOOTSTRAP = 200   # how many resamples for the uncertainty band
SEED = 11           # fixed so every run gives identical results; change for new data

# ============================================================================
# COLOURS -- appearance only, nothing statistical happens here
# ============================================================================
BLUE = '#2a78d6'
ORANGE = '#eb6834'
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRIDLINE = '#e1e0d9'
AXIS = '#c3c2b7'

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'text.color': INK,
    'axes.labelcolor': INK_2,
    'axes.edgecolor': AXIS,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'grid.color': GRIDLINE,
})


# ============================================================================
# BUILDING BLOCK 1: a fitted 1-D curve
#
# One class covers both families. A "normal" is stored as a skew normal whose
# shape dial is 0, because those are mathematically the same thing (glossary:
# THE SKEW NORMAL). That means the rest of the script never has to branch on
# which family it is holding -- and the copula in particular does not care.
# ============================================================================
class Marginal:
    """One fitted curve for one measurement. Optionally chopped off at [lo, hi].

    "Marginal" is the standard word for the curve of a single variable
    considered on its own, ignoring the other one.
    """

    def __init__(self, family, params, bounds=None):
        self.family = family          # 'skew-normal' or 'normal' (for display only)
        self.params = params          # (location, scale, shape); shape is 0.0 for a normal
        self.bounds = bounds          # (lo, hi) if truncated, or None if not

    @property
    def n_params(self):
        """How many dials this curve has. AIC charges 2 points per dial."""
        return 3 if self.family == 'skew-normal' else 2

    def _raw_cdf(self, x):
        """Running total of the UNtruncated curve (ignores self.bounds)."""
        loc, scale, shape = self.params
        return skewnorm.cdf(x, shape, loc, scale)

    def _renorm(self):
        """How much of the untruncated curve's area survives inside the bounds.

        Chopping the curve at 0 and 1 throws some area away, so what is left no
        longer sums to 1. Dividing by this number scales it back up. Returns 1.0
        (a no-op) when there are no bounds.
        """
        if self.bounds is None:
            return 1.0
        lo, hi = self.bounds
        return self._raw_cdf(hi) - self._raw_cdf(lo)

    def pdf(self, x):
        """Height of the curve at x -- how common that value is."""
        loc, scale, shape = self.params
        # Divide by the surviving area so the chopped curve still totals 1.
        out = skewnorm.pdf(x, shape, loc, scale) / self._renorm()
        if self.bounds is not None:
            # Outside the physical bounds the height is flatly zero.
            lo, hi = self.bounds
            out = np.where((x < lo) | (x > hi), 0.0, out)
        return out

    def cdf(self, x):
        """Running total at x -- the fraction of measurements below x."""
        if self.bounds is None:
            return self._raw_cdf(x)
        # Rebase so the running total is 0 at the low bound and 1 at the high one.
        lo, hi = self.bounds
        out = (self._raw_cdf(x) - self._raw_cdf(lo)) / self._renorm()
        return np.clip(out, 0.0, 1.0)

    def mean(self):
        """The average value this curve predicts.

        Note this is the average of the CURVE, not of the measured numbers.

        Without truncation there is an exact formula, so we just use it. With
        truncation no such formula exists, so we fall back on brute force: chop
        the range into 20000 slivers, add up (value x height) across them, and
        divide by the total height. Dividing cancels most of the sliver-counting
        error, and 20001 points is more than enough -- the answer stops moving
        in the 10th decimal place well before that.
        """
        loc, scale, shape = self.params
        if self.bounds is None:
            delta = shape / np.sqrt(1 + shape**2)
            return loc + scale * delta * np.sqrt(2 / np.pi)
        grid = np.linspace(self.bounds[0], self.bounds[1], 20001)
        density = self.pdf(grid)
        return np.trapezoid(grid * density, grid) / np.trapezoid(density, grid)

    def describe(self):
        """One-line human-readable summary, for printing."""
        loc, scale, shape = self.params
        text = f'{self.family}  loc={loc:+.3f}  scale={scale:.3f}'
        if self.family == 'skew-normal':
            text += f'  shape={shape:+.3f}'
        if self.bounds is not None:
            text += f'  truncated to [{self.bounds[0]:g}, {self.bounds[1]:g}]'
        return text


# ============================================================================
# BUILDING BLOCK 2: scoring a candidate curve
# ============================================================================
def loglik(x, loc, scale, shape, bounds):
    """Score how well one specific curve explains the measurements in `x`.

    Bigger = better. See the glossary entry LIKELIHOOD. Set shape=0 to score an
    ordinary normal.
    """
    # Reject impossible dial settings outright. A negative width is meaningless,
    # and the optimiser will happily try one if we let it.
    if not np.isfinite([loc, scale, shape]).all() or scale <= 0:
        return -np.inf

    # How many widths from the centre each measurement sits.
    z = (x - loc) / scale

    # One score per measurement, added up (logs, so adding == multiplying).
    # This is the standard skew-normal formula; the norm.logcdf term is what
    # produces the lean.
    total = np.sum(np.log(2) - np.log(scale) + norm.logpdf(z) + norm.logcdf(shape * z))

    if bounds is not None:
        # TRUNCATION CORRECTION, and it is not optional.
        #
        # Chopping the curve at the bounds throws away some of its area, so the
        # chopped curve is TALLER than the original everywhere inside (its area
        # still has to be 1). Subtracting n*log(surviving area) accounts for
        # that extra height.
        #
        # Leave this out and the fit goes wrong in a specific way: the optimiser
        # gets rewarded for shoving the curve's mass outside the bounds where
        # nothing has to be explained, so location and scale drift outward.
        lo, hi = bounds
        mass = skewnorm.cdf(hi, shape, loc, scale) - skewnorm.cdf(lo, shape, loc, scale)
        if mass <= 0:
            return -np.inf
        total -= len(x) * np.log(mass)

    return total if np.isfinite(total) else -np.inf


# ============================================================================
# BUILDING BLOCK 3: the search for the best dial settings
# ============================================================================
def _fit(x, bounds, free_shape, starts):
    """Hunt for the dial settings that explain `x` best.

    free_shape=False locks the shape dial at 0, which fits an ordinary normal.
    free_shape=True lets it move, which fits a skew normal.
    `starts` is the list of shape values to begin the search from.
    """
    def nll(theta):
        """What the optimiser minimises: NEGATIVE log-likelihood."""
        loc = theta[0]

        # The optimiser works in log-space for the width. Reason: it wants to
        # try any number at all, including negatives, but a width must stay
        # positive. exp() of anything is positive, so this makes an illegal
        # value impossible instead of merely rejected.
        scale = np.exp(theta[1])

        shape = np.clip(theta[2], -ALPHA_BOUND, ALPHA_BOUND) if free_shape else 0.0
        value = loglik(x, loc, scale, shape, bounds)
        # inf tells the optimiser "this direction is hopeless, back off".
        return -value if np.isfinite(value) else np.inf

    # Run the search once per starting guess and keep whichever finished lowest.
    best = None
    for shape_0 in starts:
        # Sensible opening guess: centre on the data's average, width on its spread.
        theta_0 = [x.mean(), np.log(x.std() + 1e-9), shape_0]
        res = minimize(nll, theta_0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'maxfev': 8000,
                                'xatol': 1e-9, 'fatol': 1e-11})
        if best is None or res.fun < best.fun:
            best = res

    loc, scale = best.x[0], np.exp(best.x[1])
    shape = np.clip(best.x[2], -ALPHA_BOUND, ALPHA_BOUND) if free_shape else 0.0
    return (loc, scale, shape), loglik(x, loc, scale, shape, bounds)


# Where to start the shape search from. Several starts, not one, for two reasons:
#
#   1. shape=0 is a perfectly flat spot in the landscape. An optimiser dropped
#      exactly there sees no downhill direction and never moves.
#   2. A single start can settle into a shallow dip that is not the deepest one.
#
# These straddle zero in both directions at three different magnitudes.
SHAPE_STARTS = (-6.0, -2.0, -0.5, 0.5, 2.0, 6.0)


def select_marginal(x, bounds, label, verbose=True):
    """Fit BOTH families to `x`, then keep the better one.

    Two candidates compete: an ordinary normal (2 dials) and a skew normal
    (3 dials). Lower AIC wins -- but with one extra rule, explained below.
    """
    (loc_n, scale_n, _), ll_normal = _fit(x, bounds, free_shape=False, starts=(0.0,))
    params_sn, ll_skew = _fit(x, bounds, free_shape=True, starts=SHAPE_STARTS)

    # AIC = 2*(dials) - 2*(log-likelihood). Lower is better.
    aic_normal = 2 * 2 - 2 * ll_normal
    aic_skew = 2 * 3 - 2 * ll_skew

    # THE DISQUALIFICATION RULE -- without this, the comparison is worthless.
    #
    # With few measurements the skew-normal search does not settle anywhere. It
    # keeps leaning harder and harder, off toward infinity, because a curve
    # leaning infinitely can be bent to pass through a handful of points. Left
    # unchecked it posts a great score and "wins" the AIC contest with a curve
    # that is pure nonsense.
    #
    # Measured on 40 random 10-point samples: the runaway fit won 25 times, and
    # every single one of those 25 had a shape around 1e15. So: if the shape
    # ends up jammed against the safety rail, the data never pinned it down.
    # Throw the candidate out rather than believe it.
    diverged = abs(params_sn[2]) >= ALPHA_BOUND - 0.1
    chose_skew = (not diverged) and aic_skew < aic_normal

    if verbose:
        print(f'  {label} (n = {len(x)})')
        print(f'    normal        AIC = {aic_normal:8.2f}')
        if diverged:
            print(f'    skew-normal   AIC = {aic_skew:8.2f}   DISQUALIFIED: '
                  f'shape ran to the |{ALPHA_BOUND:g}| bound, so {len(x)} points '
                  f'cannot pin it down')
        else:
            print(f'    skew-normal   AIC = {aic_skew:8.2f}   (shape = {params_sn[2]:+.2f})')
        winner = 'skew-normal' if chose_skew else 'normal'
        print(f'    -> keeping the {winner}')

    if chose_skew:
        return Marginal('skew-normal', params_sn, bounds)
    return Marginal('normal', (loc_n, scale_n, 0.0), bounds)


# ============================================================================
# BUILDING BLOCK 4: measuring how much to trust a fit (the bootstrap)
# ============================================================================
def refit_like(model, x):
    """Refit `model`'s family to a new set of numbers.

    Used only inside the bootstrap, where the family question is already
    settled, so 2 starting guesses are enough instead of 6. That keeps 200
    resamples down to a reasonable runtime.
    """
    free = model.family == 'skew-normal'
    params, _ = _fit(x, model.bounds, free_shape=free,
                     starts=(-2.0, 2.0) if free else (0.0,))
    return Marginal(model.family, params, model.bounds)


def bootstrap_band(model, x, grid, rng, n_resample=N_BOOTSTRAP):
    """How much would the fitted curve have wobbled with different luck?

    200 times over: build a fake sample by drawing from the real measurements at
    random with replacement, refit it, and record the resulting curve. Then at
    every point along the grid report the 5th and 95th percentile across those
    200 curves. That pair of lines is the shaded band on the plot, and 90% of
    the refits landed between them.

    A wide band is not a bug. It is the honest statement that this many points
    could not narrow things down further.
    """
    curves = np.empty((n_resample, grid.size))
    for i in range(n_resample):
        # replace=True is what makes it a bootstrap: some measurements get
        # picked twice, others not at all, mimicking sampling luck.
        curves[i] = refit_like(model, rng.choice(x, x.size, replace=True)).pdf(grid)
    return np.percentile(curves, [5, 95], axis=0)


# ============================================================================
# BUILDING BLOCK 5: the copula -- gluing two 1-D curves into one 2-D surface
#
#     f(x, y) = c(F1(x), F2(y)) * f1(x) * f2(y)
#
#         f1, f2  the two finished 1-D curves (heights)
#         F1, F2  their running totals, each a number from 0 to 1
#         c       the glue factor, which is where RHO acts
#
# When RHO is 0, c is exactly 1 everywhere and the formula collapses to plain
# multiplication -- the two measurements are independent. Turning RHO up makes c
# larger along one diagonal and smaller along the other, tilting the surface,
# while leaving both original curves perfectly intact.
# ============================================================================

# The running total hits exactly 0.0 and exactly 1.0 at the far edges of the
# grid. Feeding those into norm.ppf (its reverse) asks "what value is 0% of the
# data below?", whose honest answer is minus infinity. That infinity then
# poisons the arithmetic and the entire surface comes out as NaN. Nudging just
# inside the endpoints avoids it at a cost far below anything visible.
CDF_CLIP = 1e-12


def copula_joint_pdf(x_grid, y_grid, m_x, m_y, rho):
    """Build the 2-D surface from the two 1-D curves.

    Returns a grid of heights with one row per y value and one column per x.
    """
    fx, fy = m_x.pdf(x_grid), m_y.pdf(y_grid)

    if rho == 0.0:
        # Independent: the glue factor is 1, so this is just height x height.
        # Handled separately so it is exact rather than merely very close.
        return fy[:, None] * fx[None, :]

    # Translate each axis onto a common ruler: "how many standard deviations
    # out is this, if the running total were a plain bell curve?" The glue is
    # defined in terms of that shared ruler, which is how it can join two
    # completely different families without disturbing either.
    #
    # Note qx is computed on the 1-D x grid and qy on the 1-D y grid, then
    # broadcast against each other with [None, :] and [:, None]. Doing it on the
    # full 2-D mesh instead computes the same values thousands of times over and
    # is roughly 100x slower.
    qx = norm.ppf(np.clip(m_x.cdf(x_grid), CDF_CLIP, 1 - CDF_CLIP))[None, :]
    qy = norm.ppf(np.clip(m_y.cdf(y_grid), CDF_CLIP, 1 - CDF_CLIP))[:, None]

    exponent = -(rho**2 * (qx**2 + qy**2) - 2 * rho * qx * qy) / (2 * (1 - rho**2))
    c = np.exp(exponent) / np.sqrt(1 - rho**2)
    return c * fy[:, None] * fx[None, :]


def hdr_levels(Z, cell_area, masses):
    """Find the heights to draw the contour rings at.

    To find the "68% ring": sort every point on the surface from tallest to
    shortest, then walk down that list adding up area as you go. The moment the
    running total reaches 68%, the height you have reached is the ring's height.

    Working tallest-first is what makes the ring tight around the peak rather
    than some arbitrary shape that happens to enclose 68%.
    """
    Z_sorted = np.sort(Z.ravel())[::-1]
    cumsum = np.cumsum(Z_sorted) * cell_area
    cumsum /= cumsum[-1]
    return [Z_sorted[np.argmin(np.abs(cumsum - m))] for m in masses]


# ============================================================================
# STEP 1 -- Invent the two sets of measurements
# ============================================================================
rng = np.random.default_rng(SEED)

# Intensity cannot leave [0, 1], so we draw from the CHOPPED curve directly
# rather than drawing freely and discarding whatever misses.
#
# The trick uses the running total. Pick a random number between "the running
# total at 0" and "the running total at 1", then run the total backwards to see
# which intensity it corresponds to. Every draw lands inside the bounds by
# construction, first try, no waste.
lo, hi = INTENSITY_BOUNDS
t = TRUE_INTENSITY
F_lo, F_hi = skewnorm.cdf([lo, hi], t['alpha'], t['xi'], t['omega'])
intensity = skewnorm.ppf(rng.uniform(F_lo, F_hi, N_INTENSITY),
                         t['alpha'], t['xi'], t['omega'])

# The z score has no hard wall, so an ordinary draw is fine.
t = TRUE_ZSCORE
zscore = skewnorm.rvs(t['alpha'], t['xi'], t['omega'], size=N_ZSCORE,
                      random_state=rng)

print('Synthetic data')
print(f'  intensity  n={N_INTENSITY}  range [{intensity.min():.3f}, {intensity.max():.3f}]'
      f'  median {np.median(intensity):.3f}')
print(f'  z score    n={N_ZSCORE}  range [{zscore.min():+.3f}, {zscore.max():+.3f}]'
      f'  median {np.median(zscore):+.3f}')
print('  the two sets are NOT paired -- there are no (intensity, z score) observations')

# ============================================================================
# STEP 2 -- Fit each measurement separately
#
# Each bag of numbers is fitted entirely on its own. Nothing about intensity
# influences the z-score fit or the other way round. That independence is the
# point: it is what lets the copula in Step 4 do its job.
# ============================================================================
print('\nFitting each marginal separately (lower AIC wins; AIC charges 2 per parameter)')
m_intensity = select_marginal(intensity, INTENSITY_BOUNDS, 'intensity')
m_zscore = select_marginal(zscore, None, 'z score')

print('\nSelected marginals')
print(f'  intensity : {m_intensity.describe()}')
print(f'  z score   : {m_zscore.describe()}')
if m_intensity.bounds is not None:
    kept = m_intensity._renorm()
    print(f'  truncation kept {kept:.4f} of the untruncated intensity density '
          f'({100 * (1 - kept):.2f}% fell outside [0, 1] and was renormalised away)')

# ============================================================================
# STEP 3 -- The average, drawn later as a blue circle
#
# The copula never touches either curve, so the 2-D average is simply each
# curve's own average paired up. Which means it does NOT move when you change
# RHO -- changing how two things relate does not change either one's average.
# ============================================================================
mean_point = (m_intensity.mean(), m_zscore.mean())
print('\nAverage (mean of the fitted joint distribution)')
print(f'  intensity {mean_point[0]:8.4f}   (raw sample mean {intensity.mean():.4f})')
print(f'  z score   {mean_point[1]:+8.4f}   (raw sample mean {zscore.mean():+.4f})')
print('  unaffected by rho: the copula changes dependence, never the marginals')

# ============================================================================
# STEP 4 -- Glue the two curves together, then check the glue held
# ============================================================================

# The grid we will actually draw: 400 x 400 points across the plotting window.
gx = np.linspace(lo, hi, 400)
gy = np.linspace(ZSCORE_RANGE[0], ZSCORE_RANGE[1], 400)
dx, dy = gx[1] - gx[0], gy[1] - gy[0]
J = copula_joint_pdf(gx, gy, m_intensity, m_zscore, RHO)

# A SEPARATE, WIDER grid, used only for checking.
#
# Why not just check on the grid above? Because the z-score curve is not
# truncated, so a sliver of it lives outside the plotted [-3, 3]. Adding up
# only what is on screen would come to slightly less than 1 and make a
# perfectly correct surface look broken -- by exactly the amount left off
# screen. Checking out to +/-12 captures effectively all of it.
#
# Intensity needs no widening: it is genuinely zero outside [0, 1].
cx = np.linspace(lo, hi, 2400)
cy = np.linspace(-12.0, 12.0, 2400)
cdx, cdy = cx[1] - cx[0], cy[1] - cy[0]
C = copula_joint_pdf(cx, cy, m_intensity, m_zscore, RHO)

# Check 1: does the whole surface add up to 1? (It has to -- something happens.)
mass = C.sum() * cdx * cdy
# Checks 2 and 3: squash the surface flat onto each axis. The original 1-D curve
# must come back unchanged. This is the copula's defining promise, so if it
# fails, the glue is wrong.
marg_x_err = np.abs(C.sum(axis=0) * cdy - m_intensity.pdf(cx)).max()
marg_y_err = np.abs(C.sum(axis=1) * cdx - m_zscore.pdf(cy)).max()
# Check 4: with RHO=0 the surface must equal plain height x height, exactly.
product_err = np.abs(C - m_zscore.pdf(cy)[:, None] * m_intensity.pdf(cx)[None, :]).max()
# For information: how much z-score curve sits outside the plotting window.
window_tail = 1.0 - (m_zscore.cdf(ZSCORE_RANGE[1]) - m_zscore.cdf(ZSCORE_RANGE[0]))

print(f'\nJoint distribution (Gaussian copula, rho = {RHO})')
print(f'  integrates to                         {mass:.6f}   (want 1.0)')
print(f'  max error, recovered intensity marg.  {marg_x_err:.3e}   (want ~0)')
print(f'  max error, recovered z-score marg.    {marg_y_err:.3e}   (want ~0)')
print(f'  max |joint - f1*f2|                   {product_err:.3e}'
      f'   ({"exactly 0 because rho = 0" if RHO == 0 else "non-zero: rho couples them"})')
print(f'  z-score mass outside the plotted [{ZSCORE_RANGE[0]:g}, {ZSCORE_RANGE[1]:g}] '
      f'window: {window_tail:.3e}')

# ============================================================================
# STEP 5 -- How much does each sample size actually buy us?
# ============================================================================
print(f'\nBootstrapping each marginal ({N_BOOTSTRAP} resamples each)...')
band_intensity = bootstrap_band(m_intensity, intensity, gx, np.random.default_rng(SEED + 1))
band_zscore = bootstrap_band(m_zscore, zscore, gy, np.random.default_rng(SEED + 2))

# Report each band as a PERCENTAGE of its own curve's height. The two
# measurements live on completely different scales (0-to-1 versus -3-to-+3), so
# their raw band widths cannot be compared against each other. Percentages can.
for name, band, model, grid_1d in (('intensity', band_intensity, m_intensity, gx),
                                   ('z score', band_zscore, m_zscore, gy)):
    peak = model.pdf(grid_1d).max()
    width = np.ptp(band[:, band[1].argmax()])
    print(f'  {name:10s} band at its peak: {width:.3f} = {width / peak:5.1%} of peak density')

# ============================================================================
# STEP 6 -- Draw everything
#
# Layout: the 2-D surface fills the large panel; each measurement's own 1-D
# curve sits on the edge it belongs to (intensity along the top, z score down
# the right side).
# ============================================================================
MASSES = [0.50, 0.68, 0.87, 0.95]
levels = sorted(hdr_levels(J, dx * dy, MASSES))
level_labels = {lvl: f'{m:.0%}' for lvl, m in zip(levels, sorted(MASSES, reverse=True))}

fig = plt.figure(figsize=(9, 8.6))
# Fixed margins rather than bbox_inches='tight': this reserves space for the
# title block so it can never collide with the panels.
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                      wspace=0.06, hspace=0.06,
                      left=0.09, right=0.97, bottom=0.09, top=0.85)
ax_main = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)     # intensity's own curve
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)   # z score's own curve

# --- the 2-D surface, drawn as contour rings ---
GX, GY = np.meshgrid(gx, gy)
contours = ax_main.contour(GX, GY, J, levels=levels, colors=BLUE, linewidths=1.8)
ax_main.clabel(contours, fmt=level_labels, fontsize=8, colors=INK_2, inline=True)
ax_main.set_xlim(lo, hi)
ax_main.set_ylim(*ZSCORE_RANGE)
ax_main.set_xlabel('Intensity')
ax_main.set_ylabel('Z score')
ax_main.grid(True, linewidth=0.6, alpha=0.7)
ax_main.set_axisbelow(True)
for side in ('top', 'right'):
    ax_main.spines[side].set_visible(False)

# --- the average, as a blue circle ---
# The pale ring around it is the page colour, which keeps the marker readable
# where it happens to land on top of a contour line.
ax_main.plot(mean_point[0], mean_point[1], marker='o', color=BLUE, markersize=10,
             markeredgecolor=SURFACE, markeredgewidth=2, linestyle='none', zorder=6)
ax_main.annotate(f'average\n({mean_point[0]:.2f}, {mean_point[1]:+.2f})',
                 xy=mean_point, xytext=(14, 8), textcoords='offset points',
                 fontsize=8.5, color=INK_2, zorder=6,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=SURFACE,
                           edgecolor='none', alpha=0.85))

# There are deliberately no dots scattered across this panel. A dot would need
# an intensity AND a z score to position it, and no such pairing exists in the
# data. Better to say so than to leave a reader wondering.
ax_main.text(0.5, 0.02,
             'No scatter: the two samples are unpaired, so no (intensity, z score) point exists',
             transform=ax_main.transAxes, ha='center', va='bottom',
             fontsize=8.5, color=MUTED)
ax_main.legend(
    handles=[
        Line2D([], [], color=BLUE, lw=1.8, label=f'Joint density (copula, rho = {RHO})'),
        Line2D([], [], color=BLUE, lw=0, marker='o', markersize=8,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label='Average'),
        Patch(facecolor=BLUE, alpha=0.18, edgecolor='none', label='90% bootstrap band'),
        Patch(facecolor=GRIDLINE, edgecolor='none', label='Histogram of measurements'),
        Line2D([], [], color=ORANGE, lw=0, marker='|', markersize=7,
               markeredgewidth=1.1, label='Individual measurements'),
    ],
    loc='upper right', frameon=True, facecolor=SURFACE, edgecolor=GRIDLINE,
    framealpha=0.92, fontsize=9, labelcolor=INK_2,
)

# --- the two edge panels ---
# Same recipe both times, just rotated: grey histogram of the measurements,
# shaded bootstrap band, fitted curve on top, and orange ticks marking every
# individual measurement.
panels = (
    (ax_top, 'v', intensity, gx, m_intensity, band_intensity),
    (ax_right, 'h', zscore, gy, m_zscore, band_zscore),
)
for panel, orient, data, grid_1d, model, band in panels:
    curve = model.pdf(grid_1d)
    n_bins = max(8, min(24, data.size // 2))
    orientation = 'vertical' if orient == 'v' else 'horizontal'
    heights, _, _ = panel.hist(data, bins=n_bins, density=True, color=GRIDLINE,
                               edgecolor=SURFACE, linewidth=0.8,
                               orientation=orientation)
    # Make room for whichever is tallest -- curve, band, or histogram bar --
    # otherwise the bars get clipped by the panel edge.
    top = max(curve.max(), band[1].max(), heights.max()) * 1.10
    rug = -0.04 * top   # tick marks sit just below the baseline
    if orient == 'v':
        panel.fill_between(grid_1d, band[0], band[1], color=BLUE, alpha=0.18, linewidth=0)
        panel.plot(grid_1d, curve, color=BLUE, lw=1.8)
        panel.plot(data, np.full(data.size, rug), '|',
                   color=ORANGE, markersize=6, markeredgewidth=1.1)
        panel.tick_params(labelbottom=False, left=False, labelleft=False)
        panel.set_ylim(-0.09 * top, top)
    else:
        panel.fill_betweenx(grid_1d, band[0], band[1], color=BLUE, alpha=0.18, linewidth=0)
        panel.plot(curve, grid_1d, color=BLUE, lw=1.8)
        panel.plot(np.full(data.size, rug), data, '_',
                   color=ORANGE, markersize=6, markeredgewidth=1.1)
        panel.tick_params(labelleft=False, bottom=False, labelbottom=False)
        panel.set_xlim(-0.09 * top, top)
    for side in ('top', 'right', 'left' if orient == 'v' else 'bottom'):
        panel.spines[side].set_visible(False)

# --- titles and captions ---
fig.text(0.09, 0.945, 'Intensity and z score joined by a Gaussian copula',
         color=INK, fontsize=14, fontweight='bold', ha='left', va='center')
subtitle = (f'intensity: {m_intensity.family}, n = {N_INTENSITY}      '
            f'z score: {m_zscore.family}, n = {N_ZSCORE}      '
            f'dependence: rho = {RHO}')
fig.text(0.09, 0.905, subtitle, color=INK_2, fontsize=9.5, ha='left', va='center')
fig.text(0.09, 0.028,
         'Contours hold 50 / 68 / 87 / 95% of the mass. Orange ticks are the actual '
         'measurements; shading is the 90% bootstrap band.',
         color=MUTED, fontsize=8.5, ha='left', va='center')

out_path = OUT_DIR / 'copula_joint_fit.png'
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f'\nSaved {out_path}')
