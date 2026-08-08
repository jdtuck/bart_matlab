# BART for MATLAB

Bayesian Additive Regression Trees with flexible nonparametric modeling of
covariates for **continuous**, **binary**, **categorical** and **time-to-event**
outcomes. A from-scratch MATLAB implementation of the models described in

> Sparapani, R., Spanbauer, C. and McCulloch, R. (2021). *Nonparametric Machine
> Learning and Efficient Computation with Bayesian Additive Regression Trees:
> The BART R Package.* Journal of Statistical Software, 97(1), 1–66.
> [doi:10.18637/jss.v097.i01](https://doi.org/10.18637/jss.v097.i01)

No toolboxes required — the gamma, truncated-normal and Pólya–Gamma samplers,
the normal CDF/quantile functions and the percentile utilities are all included.
Runs unmodified in MATLAB (R2016b or later) and in GNU Octave 7+.

---

## Quick start

Put every `.m` file on your path, then:

```matlab
% ---- continuous outcome -------------------------------------------------
n = 500; p = 10;
x = rand(n, p);
y = 10*sin(pi*x(:,1).*x(:,2)) + 20*(x(:,3)-0.5).^2 + 10*x(:,4) + 5*x(:,5) + randn(n,1);

fit = wbart(x, y, rand(200, p));          % train, and predict at 200 new points
fit.yhat_test_mean                        % posterior mean of f at the test points
mean(fit.sigma)                            % posterior mean residual sd
fit.varprob                                % share of splits used by each covariate

% ---- binary outcome -----------------------------------------------------
opts = bartOptions('ntree', 50);           % 50 trees is the usual choice here
fit = pbart(x, y > median(y), [], opts);
fit.prob_train_mean                        % posterior mean P(Y = 1 | x)

% ---- categorical outcome ------------------------------------------------
yc  = 1 + (x(:,1) > 0.4) + (x(:,2) > 0.7); % three levels
fit = mbart(x, yc, [], opts);
fit.prob_train_mean                        % n-by-J matrix of probabilities
fit.pred_train                             % most probable level per row

% ---- time-to-event outcome ----------------------------------------------
times = -log(rand(n,1)) ./ (0.5*exp(1.5*x(:,1)));
cens  = 0.5 + 3*rand(n,1);
fit   = survbart(min(times,cens), times <= cens, x, [0.1 0.5; 0.9 0.5], ...
                 bartOptions('ntree', 50, 'K', 25));
fit.times                                  % the time grid
fit.surv_mean                              % npred-by-K posterior mean S(t|x)
[fit.surv_lower; fit.surv_upper]            % 95% pointwise credible bands
```

Run `demo_bart` for a worked example of all four outcome types against known
truth, and `test_bart` for the correctness checks.

---

## Files

| File | Role |
|---|---|
| `wbart.m` | continuous outcome, Gaussian errors |
| `pbart.m` | binary outcome, probit link (Albert–Chib augmentation) |
| `lbart.m` | binary outcome, logistic link (Pólya–Gamma augmentation) |
| `mbart.m` | categorical outcome, J ≥ 2 unordered levels |
| `survbart.m` | right-censored time-to-event outcome |
| `bartCore.m` | the shared Bayesian backfitting MCMC engine |
| `bartPredict.m` | posterior prediction at new covariates from stored trees |
| `bartOptions.m` | all tuning parameters and priors, with defaults |
| `bartModelMatrix.m` | numeric design matrix, dummy-coding factors |
| `bartMakeCuts.m`, `bartBinX.m`, `bartAssign.m` | cutpoint grid, binning, vectorized tree descent |
| `bartGamrnd.m`, `bartTruncNormRnd.m`, `bartPolyaGammaRnd.m` | samplers |
| `bartPhi.m`, `bartPhiInv.m`, `bartPctl.m`, `corrPearson.m` | small numeric utilities |
| `demo_bart.m`, `test_bart.m`, `argmaxrow.m` | demonstration and tests |

---

## The model

Every outcome type reduces to one weighted Gaussian sum-of-trees problem, which
is what `bartCore` samples:

```
y_i = sum_{t=1..m} g(x_i; T_t, M_t) + eps_i,     eps_i ~ N(0, sigma^2 / w_i)
```

with the Chipman, George & McCulloch (2010) priors:

- **Tree structure.** A node at depth `d` is internal with probability
  `alpha*(1+d)^(-beta)`, defaults `alpha = 0.95`, `beta = 2`. This keeps
  individual trees shallow so that the ensemble is a sum of weak learners.
- **Split rules.** A variable is drawn uniformly (or from Dirichlet
  probabilities when `sparse = true`), then a cutpoint uniformly from that
  variable's grid of at most `numcut` candidates.
- **Leaf means.** Independent `N(0, tau^2)`. Because this is conjugate, the leaf
  means are integrated out of the Metropolis–Hastings ratio analytically and
  then drawn exactly, which is what makes the sampler mix acceptably.
- **Error variance.** `sigma^2 ~ nu*lambda / chi^2_nu`, with `lambda` solved so
  that `P(sigma < sigest) = q`; `sigest` is the residual sd of a least-squares
  pilot fit (or `sd(y)` when `p >= n`). Continuous outcomes only.

Trees are updated one at a time on the partial residuals (Bayesian
backfitting). Three Metropolis–Hastings moves are used:

- **GROW** — split a randomly chosen leaf
- **PRUNE** — collapse a randomly chosen internal node whose children are both leaves
- **CHANGE** — repropose the split rule at such a node

The GROW/PRUNE transition ratios include the correct reverse-move counts and the
boundary case where the current tree is a single stump. Proposals that would
leave an empty child are rejected, which is the standard simplification in place
of enumerating only the split rules that remain available given a node's
ancestors.

### Priors by outcome type

| Model | Leaf prior sd `tau` | Offset | `sigma` |
|---|---|---|---|
| `wbart` | `range(y) / (2*k*sqrt(ntree))` | `mean(y)` | sampled |
| `pbart` | `3 / (k*sqrt(ntree))` | `Phi^{-1}(mean(y))` | fixed at 1 |
| `lbart` | `4 / (k*sqrt(ntree))` | `logit(mean(y))` | fixed at 1 |

`k` (default 2) is the shrinkage dial: larger `k` shrinks the fit harder toward
the offset.

### How each outcome type maps onto the engine

**Continuous.** Direct. Observation weights `w` are supported, so
`var(y_i) = sigma^2 / w_i`.

**Binary, probit.** Albert & Chib (1993) augmentation: draw latent
`z_i ~ N(mu0 + f(x_i), 1)` truncated positive when `y_i = 1` and negative when
`y_i = 0`. Given `z`, the problem is Gaussian with `sigma = 1`. The truncated
draws use plain rejection near the mode and Robert's (1995) exponential-proposal
method in the tails, so they stay exact even when fitted probabilities approach
0 or 1.

**Binary, logit.** Pólya–Gamma augmentation (Polson, Scott & Windle, 2013): draw
`omega_i ~ PG(1, mu0 + f(x_i))`, after which `(y_i - 1/2)/omega_i` is Gaussian
around the linear predictor with variance `1/omega_i` — exactly the weighted
problem the engine already solves.

**Categorical.** The multinomial likelihood factors into conditional binary
problems,

```
P(Y = j | Y >= j) = p_j(x),   j = 1, ..., J-1
P(Y = j)          = p_j(x) * prod_{l < j} (1 - p_l(x))
P(Y = J)          = prod_{l < J} (1 - p_l(x))
```

so each `p_j` is fitted by an independent probit BART on the rows still at risk
(`y >= j`). Because the factorization is exact, running the fits separately and
multiplying the draws together **draw by draw** yields a genuine joint posterior
sample of the category probabilities, not just point estimates. Each conditional
fit predicts at every training and test point so the recombination is possible.

**Time-to-event.** The person-period (grouped hazard) representation: follow-up
is cut at the distinct event times `t_1 < ... < t_K`, and each subject
contributes one Bernoulli row per interval in which it is still at risk, with

```
P(event in interval k | at risk) = p(t_k, x) = Phi(mu0 + f(t_k, x))
S(t_k | x) = prod_{l <= k} (1 - p(t_l, x))
```

Since the survival likelihood factors into these Bernoulli terms, the whole
problem is a single probit BART on the long data set **with time as an ordinary
covariate in the first column**. The trees therefore learn the baseline hazard
and any time-by-covariate interaction from the data: there is no
proportional-hazards assumption and no parametric baseline. `opts.K` coarsens
the grid to `K` quantiles of the distinct event times, which is the main cost
control since the long data set has `sum(k_i)` rows.

### Sparse variable selection

With `opts.sparse = true`, splitting variables are drawn from
`s ~ Dirichlet(theta/p + counts)` instead of uniformly (Linero, 2018), which
concentrates splits on relevant covariates in high dimensions. Leaving
`opts.theta = 0` samples the concentration parameter from its
`theta/(theta+rho) ~ Beta(a, b)` prior on a grid.

---

## Options

All set through `bartOptions`, which takes a struct, name/value pairs, or both:

```matlab
opts = bartOptions('ntree', 50, 'ndpost', 2000, 'nskip', 500, 'sparse', true);
opts = bartOptions(opts, 'k', 3);          % modify an existing struct
```

| Option | Default | Meaning |
|---|---|---|
| `ntree` | 200 | trees in the ensemble (use 50 for `pbart`/`lbart`/`mbart`/`survbart`) |
| `ndpost`, `nskip`, `keepevery` | 1000, 100, 1 | draws kept, burn-in, thinning |
| `numcut`, `usequants` | 100, false | cutpoints per covariate; uniform or quantile grid |
| `alpha`, `beta` | 0.95, 2 | tree-depth prior |
| `k` | 2 | leaf shrinkage |
| `nu`, `q`, `sigest` | 3, 0.90, `[]` | `sigma^2` prior (continuous only) |
| `w` | `[]` | observation weights, `var(y_i) = sigma^2/w_i` |
| `sparse`, `theta`, `a`, `b`, `rho` | false, 0, 0.5, 1, `[]` | Dirichlet split prior |
| `pg`, `pp`, `pc` | 0.35, 0.35, 0.30 | GROW / PRUNE / CHANGE proposal probabilities |
| `K` | `[]` | `survbart` time-grid size; `[]` uses every distinct event time |
| `pgterms` | 100 | terms in the `lbart` Pólya–Gamma series |
| `savetrees`, `savedraws` | true, true | keep trees (needed by `bartPredict`) and full draws |
| `printevery`, `seed` | 100, `[]` | progress reporting; RNG seed |

Categorical covariates need dummy coding first — `bartModelMatrix` does this and
returns an `info` struct so new data are encoded identically:

```matlab
[X, info] = bartModelMatrix({ age, {'low','high','low', ...} });
Xnew      = bartModelMatrix({ age_new, group_new }, info);
```

BART does not need a reference level dropped, since splits are on individual
dummies.

---

## Output

Every fit returns posterior **draws**, not just point estimates, so any
functional of the fit gets a credible interval for free:

```matlab
fit = wbart(x, y, xtest);
lo  = bartPctl(fit.yhat_test, 2.5);        % pointwise 95% band for f
hi  = bartPctl(fit.yhat_test, 97.5);
```

Common fields: `yhat_train`/`yhat_test` (draws on the linear or latent scale)
and their `_mean` versions, `prob_*` for binary and categorical models,
`surv_mean`/`surv_lower`/`surv_upper` for survival, `sigma` draws for `wbart`,
`varcount` and `varprob` for variable importance, and `accept` (GROW/PRUNE/CHANGE
acceptance rates — a useful diagnostic; very low rates suggest `k` or `numcut`
needs attention).

`bartPredict(fit, xnew)` evaluates the stored posterior trees at new covariates
and reproduces in-sample test predictions exactly.

---

## Validation

`test_bart.m` checks each piece. Selected results:

- **Samplers.** Gamma(3,1) mean/variance 3.02/3.05; Gamma(0.4,1) 0.400/0.417.
  Truncated normals match their analytic means, including the deep-tail case
  `N(-3,1)` on `(0,∞)` (0.284 vs 0.283). `E[PG(1,2)]` 0.1894 vs the target
  0.1904; `E[PG(1,0)]` 0.2482 vs 0.25.
- **Continuous.** Friedman's function, `n = 500`, `p = 10`, `sigma = 1`:
  out-of-sample RMSE for `f` of 0.98 against `sd(f) ≈ 5.0`; `sigma` recovered at
  1.09 with 95% CI [1.00, 1.20]; the five relevant covariates take ~83% of
  splits. With `sparse = true` on 20 covariates where only two matter, the top
  two ranked variables are the correct ones.
- **Binary.** RMSE 0.078 for `P(Y=1|x)`, correlation 0.964 with the truth, and a
  0.190 error rate against a 0.220 Bayes rate on that draw.
- **Categorical.** Mean absolute error 0.048 on the category probabilities;
  probabilities sum to 1 to 1.3e-15; 0.849 agreement with the Bayes classifier.
- **Survival.** On a null model (no covariate effect) against Kaplan–Meier and
  the true exponential curve, BART's mean absolute error was 0.010 versus KM's
  0.008 — the machinery is calibrated against the nonparametric benchmark. With
  a covariate effect, survival curves are monotone by construction and credible
  bands are correctly ordered.
- **Prediction.** `bartPredict` reproduces stored test fits to 0.00e+00, and
  seeded runs are exactly reproducible.

---

## Practical notes and limitations

- **Speed.** This is readable reference code, not a MEX port. Roughly 45 s for
  `n = 500`, `ntree = 50`, 750 iterations in Octave; MATLAB is faster. Cost
  scales as `ntree × iterations × n`. For large problems reduce `ntree`, use
  `opts.K` for survival, and consider `numcut = 20`.
- **Interval coverage.** At reduced settings (few trees, short chains) pointwise
  95% intervals for `f` can undercover — 0.82 in the demo. The defaults
  (`ntree = 200`, longer burn-in) do better. Always check `accept` and the
  `sigma` trace before trusting intervals.
- **Survival tails.** Late-time survival is shrunk toward the overall offset
  where few subjects remain at risk, so posterior means there are biased toward
  the middle even though the credible bands still cover. This is expected BART
  behavior, not an implementation artifact — read the bands, not just the means.
- **`lbart` is approximate.** The Pólya–Gamma draws use a truncated series with a
  tail-mean correction rather than the exact Devroye method the R package uses;
  the bias is `O(K^-2)` in the mean and negligible at `pgterms = 100`, but
  `pbart` is exact and is the better default for binary outcomes.
- **Not implemented.** Recurring-event and competing-risks survival, random
  effects (`rbart`), monotone constraints, cross-validation helpers, and the
  package's convergence-diagnostic plots. The tree-rotation proposal is also
  omitted; GROW/PRUNE/CHANGE is sufficient for mixing at these tree depths.
- **Missing data** in covariates are not handled — impute or drop beforehand.

---

## References

- Sparapani, Spanbauer & McCulloch (2021). *The BART R Package.* JSS 97(1).
  [doi:10.18637/jss.v097.i01](https://doi.org/10.18637/jss.v097.i01)
- Chipman, George & McCulloch (2010). *BART: Bayesian Additive Regression Trees.*
  Annals of Applied Statistics 4(1), 266–298.
- Sparapani et al. (2016). *Nonparametric survival analysis using Bayesian
  additive regression trees.* Statistics in Medicine 35(16), 2741–2753.
- Albert & Chib (1993). *Bayesian analysis of binary and polychotomous response
  data.* JASA 88(422), 669–679.
- Polson, Scott & Windle (2013). *Bayesian inference for logistic models using
  Pólya–Gamma latent variables.* JASA 108(504), 1339–1349.
- Linero (2018). *Bayesian regression trees for high-dimensional prediction and
  variable selection.* JASA 113(522), 626–636.
- Robert (1995). *Simulation of truncated normal variables.* Statistics and
  Computing 5, 121–125.
- Marsaglia & Tsang (2000). *A simple method for generating gamma variables.*
  ACM TOMS 26(3), 363–372.
