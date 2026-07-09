---
name: math-auditor
description: >
  Run an adversarial mathematical-accuracy review of a Julia package's src/ and
  test/ directories, producing a dated markdown report plus concise, self-contained
  fix-prompt markdowns suitable for handing to a smaller model (e.g. Sonnet) in a
  later session. Use this skill whenever the user asks for an adversarial review,
  a math audit, a full code review focused on mathematical/numerical correctness,
  a check of algorithmic consistency with the literature, or says things like
  "review the math in src/", "is the algebra right?", "audit the fit/predict
  equations", "check the random feature maps / ridge regression for bugs", or
  "construct a code review as markdown". Trigger even when the user does not say
  "audit" — any request for a correctness-focused sweep of a scientific Julia
  codebase qualifies.
---

# Math Audit

Adversarial review of a scientific Julia package for **mathematical accuracy and
consistency** — not software architecture (flag architecture only when it causes
mathematical wrongness, e.g. mutation aliasing, accidental type demotion, or
inconsistent conventions between modules).

The output is written for the package's own developers: findings must cite exact
`file:line`, state the correct mathematics, and give a concrete failure scenario.
A finding that can't survive an attempt at refutation doesn't ship.

## What "adversarial" means here

Each reviewer's job is to *break* the code, not describe it. Concretely, hunt for:

- **Wrong equations**: the random feature map definitions (Fourier features
  `sigma * cos(xi . x + bias)` and neuron/activation features via
  `ScalarFunctions.jl`), the ridge-regression normal equations solved in
  `Methods.fit` (`(PhiTλinvPhi + I) beta = PhiTλinvY`), and the predictive
  mean/covariance formulas in `predictive_mean!`/`predictive_cov!` — derive the
  correct expression independently (the covariance code cites Bishop & Nasrabadi
  2006; check the implementation actually matches that derivation) and diff it
  against the code.
- **Convention drift**: rows-vs-columns for inputs (`input_dim × n_data`) versus
  features (`n_samples × output_dim × n_features`); where `sigma` scaling and the
  bias term enter relative to the nonlinearity; whether `λinv` is the noise
  covariance or its inverse and whether it is applied as `UniformScaling` or a
  full matrix consistently; whether the `1/n_features` normalization is applied
  the same number of times on the `fit` side (`PhiTλinvPhi`, `PhiTλinvY`) as on
  the `predict` side (`mean_store`, `cov_store`) — a bug here would silently
  rescale predictions. Also check the **duplicated `tullio`-threaded vs
  non-threaded code branches** in `Methods.jl` and `Utilities.jl` compute
  identical results (they are separate hand-written implementations of the same
  contraction, and drift between them is a classic bug source), and that
  `ScalarFeatures.jl` and `VectorFeatures.jl` agree on the same `build_features`
  / `predictive_mean` / `predictive_cov` contract (the scalar case is the
  `output_dim == 1` degeneracy of the vector case and must produce numerically
  identical results).
- **Statistical validity**: does the identity added in `fit` (`PhiTλinvPhi[i,i]
  += 1.0`) correspond to the ridge/Bayesian-prior regularization the docs claim,
  and is it added *before or after* the `1/n_features` normalization consistently
  with how the same normalization is (or isn't) applied to `PhiTλinvY`? Do the
  scalar and vector feature variants agree in expectation for the same
  underlying kernel? Does `FeatureSampler`'s joint distribution over `"xi"` and
  `"bias"` (via `combine_distributions`) sample and reconstruct the parameters
  in the order `build_features` expects, and does
  `transform_unconstrained_to_constrained` preserve the marginal each parameter
  was declared with?
- **Numerical soundness**: unguarded `inv`/`\`/`pinv`/`cholesky` on possibly
  singular `PhiTλinvPhi`; whether `posdef_correct`/the `Decomposition`
  constructor's automatic fallback to a nudged positive-definite matrix silently
  changes results it shouldn't; loss of symmetry/PSD-ness in `cov_store`;
  `sqrt`/`eigvals` of negative-by-roundoff quantities; whether `Factor` vs
  `PseInv` decompositions in `linear_solve` are mathematically interchangeable
  for the same input.
- **Edge cases the math must survive**: a single training point or a single
  feature (`n_features == 1`), `output_dim == 1` (scalar/matrix degeneracy),
  an unbiased sampler (`bias_distribution === nothing`), `batch_size == 0` in
  `batch_generator` (returns the whole array as one batch — check every caller
  agrees on what that means), and a `PhiTλinvPhi` that needs the
  `posdef_correct` nugget to factorize.
- **Test-math consistency**: do the tests in `test/Methods`, `test/Features`,
  `test/Samplers`, and `test/Utilities` actually pin the mathematics (recovering
  a known ridge-regression solution, an analytic kernel approximation, an
  invariant of the sampled distribution), or just check shapes and "it runs"?
  A wrong equation whose test only checks `size()` is a *double* finding: the
  bug and the missing test. Hunt specifically for test-suite pathologies that
  let wrong math ship green — they are cheap greps with outsized yield:
  a comparison expression missing its `@test` macro (computed and discarded),
  a function asserted only against *itself* (a second call via a different
  entry point re-running the same formula), clamping like `max.(0, cov)`
  applied *before* an assertion (hides negative variances), and quantities
  computed in the test but never asserted.
- **Dead numerical controls**: configuration or keywords that claim to control
  the computation but silently no-op — a kwarg immediately shadowed by a struct
  field, a batch-size/precision/threading option fetched into a local and never
  used, docstrings advertising behaviour the code no longer has. These are in
  scope even though they smell architectural: a control that silently does
  nothing is a statistical trap (users believe they bounded memory or disabled
  threading) and often marks a refactor that removed the loop but kept the
  config. When a finding looks like refactor remnants, check `git log`/`git
  blame` on the lines — history often confirms the removal, dates the bug's
  introduction, and sharpens the failure scenario.

## Workflow

### 1. Partition

List `src/*.jl` and `test/**/*.jl` with line counts. Group into review units of
roughly comparable size, pairing each source module with the tests that exercise
it. In this package the natural grouping is:

- **Core regression** — `Methods.jl` (the `RandomFeatureMethod`/`Fit`
  fit/predict/predictive_mean/predictive_cov machinery) with
  `test/Methods/runtests.jl`. This is the largest and most math-critical unit;
  split it into its own review (fit vs. predict) if it's too large for one
  agent.
- **Feature maps** — `Features.jl`, `ScalarFeatures.jl`, `VectorFeatures.jl`,
  and `ScalarFunctions.jl` with `test/Features/runtests.jl`, grouped together
  since the scalar and vector variants must agree on the same feature-map
  convention and the scalar-function nonlinearities feed directly into
  `build_features`.
- **Sampling** — `Samplers.jl` with `test/Samplers/runtests.jl` (the joint
  parameter/bias distribution and constrained/unconstrained transforms that
  feed the feature maps above).
- **Linear algebra utilities** — `Utilities.jl` (`Decomposition`,
  `linear_solve`, `posdef_correct`, `batch_generator`) with
  `test/Utilities/runtests.jl`, since a convention mismatch here (e.g. `Factor`
  vs `PseInv` producing different results) propagates silently into every
  `fit`/`predict` call.

Skip `show.jl` and `ErrorMessages.jl` from the mathematical review — they are
display/diagnostics code with no numerical content.

### 2. Fan out reviewers (parallel agents)

Before dispatching, skim the key source files yourself (10–15 minutes of
reading) and seed each agent prompt with **unit-specific attack hypotheses** —
concrete lines to check, not just the generic list. "Check the UniformScaling
shortcut `Phi * λinv.λ` against the tullio branch's index order" gets a sharper
review than "check for convention drift". In the audits run so far, the
highest-value findings traced back to seeded hypotheses; the generic list alone
tends to produce descriptions rather than attacks.

Spawn one agent per unit, in a single message so they run concurrently. Each
agent prompt must include:

- the exact file list for its unit,
- the "What adversarial means here" hunting list above (copy it in — agents
  don't see this skill),
- instructions to check code against docstrings/comments *and* against the
  standard form of the algorithm from the literature (e.g. Bishop & Nasrabadi
  2006 for the Bayesian linear regression predictive covariance that
  `predictive_cov!` cites),
- a required output format: a JSON-like list of findings, each with
  `file`, `line`, `severity` (critical / major / minor / hygiene),
  `claim` (one sentence), `evidence` (the code vs the correct math),
  `failure_scenario` (concrete inputs → wrong output),
  `verified` (`numerical` / `inspection`), and
  `suggested_fix` (optional, a few lines).

Tell agents explicitly:

- "Prefer few, well-evidenced findings over many speculative ones — but do
  report genuine minor inconsistencies. If the module's math is correct, say so
  and note the strongest invariants the tests pin."
- "When a finding concerns a fixed point, a statistical scaling, or a crash,
  verify it numerically in a scratch script if cheap — e.g. fit a
  `RandomFeatureMethod` on a tiny linear-Gaussian toy problem and compare
  `predictive_mean`/`predictive_cov` against the closed-form ridge-regression
  solution, or Monte-Carlo the sampled feature distribution against its
  analytic moments — and tag it `verified: numerical`. Numerically verified
  findings are worth far more than inspection-only ones."
- "Before claiming anything is 'silent' or 'has no warning/guard', grep for
  `@warn`, `@error`, and `throw` at the *constructors and call sites* of the
  code path, not just the function you are reading — guards often live at
  construction time (e.g. `Decomposition`'s automatic `posdef_correct` fallback)."

As each agent's report arrives, save its raw findings verbatim to a scratchpad
file (one per unit). A full audit plus verification is long enough that
context summarization mid-run can silently lose findings; the scratchpad files
are the durable record the report is assembled from.

### 3. Verify

For each critical/major finding, attempt refutation before it enters the
report: re-read the cited lines yourself, re-derive the math, and check whether
a test or an upstream transformation already accounts for it (common false
positives: a `reshape`/`permutedims` hidden in a helper, normalization done at
construction time, a convention documented elsewhere, a `@warn` at the
constructor that the reviewer never read — re-run the warn/throw grep yourself
for any "silent" claim). Prioritise findings tagged `verified: inspection`;
numerically verified ones usually need only a sanity re-read. Spawn skeptic
agents for findings you can't settle from the main context. Demote or drop
findings that don't survive; mark surviving ones **CONFIRMED** vs **PLAUSIBLE**
(couldn't fully verify). Keep a count of dropped/demoted claims and state it in
the report's method section — it tells readers the verification pass had teeth.

Expect **cross-module duplicates**: units that read each other's files (as they
should) will sometimes report the same bug from both sides — e.g. the producer
and the consumer of a broken interface. Merge them into one finding anchored at
the module that owns the fix, and cite the independent reproductions as
strengthened evidence rather than listing the bug twice.

Then build a small **conventions matrix** from the unit reports before writing
anything: rows = modules, columns = the conventions the units commented on
(where `1/n_features` normalization is applied — once in `fit`, once in
`predict`, or both; `tullio`-threaded vs non-threaded numerical equivalence;
where `sigma`/bias enter relative to the nonlinearity; rows-vs-columns for
inputs vs. features; `Factor` vs `PseInv` decomposition equivalence). Any
mismatched cell between modules that must agree is a finding candidate in
itself — in practice the worst bugs are a scale factor or transform applied on
one side of a fit/predict pair (or one feature variant) but not its sibling,
and they only become visible side by side.

### 4. Write the report

Create `full-code-review/<YYYY-MM-DD>/` (date from `date +%F`, never from
memory). Write `review.md`:

```markdown
# Adversarial Mathematical Review — <Package> (<date>)
## Scope and method          <!-- files covered, units, verification policy -->
## Summary table             <!-- ID | severity | verdict | file:line | one-line claim -->
## Critical findings         <!-- full detail: evidence, math, failure scenario, fix sketch -->
## Major findings
## Minor findings & hygiene  <!-- terser -->
## Cross-module consistency notes
## Test-coverage gaps        <!-- where math is unpinned by tests -->
## What was checked and found sound   <!-- credit where due; prevents re-auditing -->
```

Findings get stable IDs used everywhere, including fix prompts: `C1…` critical,
`M1…` major, `m1…` minor, `h1…` hygiene; grouped-minor fix prompts get `G1…`.

### 5. Write fix prompts

For each finding with an actionable fix (usually critical + major, plus grouped
minors), write `full-code-review/<date>/fix-prompts/<ID>-<slug>.md`. These are
consumed by a *smaller model in a fresh session with no context*, so each must
be self-contained:

```markdown
# Fix <ID>: <one-line title>
**File**: `src/Foo.jl`, function `bar!`, around line NNN.
**Problem**: <2–4 sentences: what the code does vs what the math requires.
Include the incorrect snippet verbatim.>
**Required change**: <exact edit, or precise description with the correct formula>
**Do not**: <guardrails — e.g. "do not change the API", "do not touch other methods">
**Verify**: <the test to run or add, with the invariant it should pin>
```

Keep each under ~40 lines. One finding per file; group only truly mechanical
repeats (e.g. the same typo pattern in five docstrings) into one prompt.
Quote enough of the offending snippet that the fixer can locate it by function
name + snippet — line numbers drift between the audit and the fix session, so
present them as hints, not anchors.
Also write `fix-prompts/README.md` organising prompts into **rounds**:
round 1 = fixes touching disjoint files (safe to apply in any order or in
parallel), round 2 = same-file prompts in an explicit sequence (state *why* —
e.g. "touches the signature region edited by C1"), round 3 = test and
docstring prompts last, so assertions and prose match the final code and any
test that *requires* a src fix (e.g. a PSD assertion that fails until the
covariance bug is fixed) lands after it. The README must also name which fixes
*intentionally change numerical results* — so the fixer checks a failing loose
regression test against the analytic reference in the prompt before "fixing"
the test — and list any maintainer decisions the prompts deliberately do not
make (restore-vs-delete a dead feature, choice of prior convention).

### 6. Report back

Final message: lead with the headline (how many confirmed critical/major
findings and the single worst one), then the report path, then a compact
summary table. Do not paste the whole report into the chat.

## Calibration

- Severity: **critical** = produces mathematically wrong results in mainstream
  use; **major** = wrong in common configurations or silently degrades
  statistical properties; **minor** = wrong in edge cases, misleading docs
  math, dead/misnamed math; **hygiene** = style-level (only if math-adjacent).
- **Loud failures cap at major.** A crash with a stack trace — however
  mainstream the path — announces itself; critical is reserved for *silently*
  wrong numbers. Reviewer agents disagree on exactly this distinction (the same
  crash has been rated critical by one unit and major by another), so apply the
  rule yourself at synthesis rather than inheriting either agent's label.
- A docstring–code mismatch is a real finding even when the code is right —
  users implement against docstrings.
- A statistically unjustified combination that the package explicitly warns
  about (e.g. a constructor `@warn "... experimental ..."`) caps at **minor**
  unless the warning itself is wrong — the trap isn't silent.
- Don't pad. If a unit is sound, the report says so in one paragraph; an audit
  that cries wolf gets ignored next time.

## Improving this skill

After delivering the report, offer: "Would you like to improve the
**math-auditor** skill itself using skill-creator? You can share suggestions, or
I can analyse this run — finding quality, false-positive rate, fix-prompt
usability — to refine the skill for next time."
