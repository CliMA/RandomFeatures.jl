---
name: error-message-manager
description: >
  Rewrite vague, delayed, or low-context Julia error messages into structured,
  actionable diagnostics. Invoke this skill whenever the user mentions: error
  message, improve errors, rewrite @assert, ArgumentError, DimensionMismatch,
  DomainError, vague error, error rewrite, Julia exception, diagnostic, throw,
  validation, early check, assert to throw, loop context, catch and rethrow,
  warn string, or asks to improve how the code fails. Also use it when reviewing
  code for user-facing clarity, when a user says errors are confusing or
  unhelpful, or when auditing a module for low-context exceptions. Use it
  proactively when you see bare @assert, error("..."), throw(ErrorException(...)),
  @warn string(...), or catch blocks that do not include the original exception
  in their re-throw in Julia code you are reading or editing.
---

# error-message-manager

Rewrite vague, delayed, or low-context Julia error messages into structured,
actionable diagnostics. The goal is errors that tell the user exactly what went
wrong, what was expected, what was received, and—whenever a likely fix exists—
what to do next. Prefer catching mistakes early (at API boundaries) over letting
them propagate into cryptic numerical failures.

## Workflow

### Step 0 — Offer an Explore agent for multi-file scope

If the user's request covers more than one file — a whole directory, a module, or
the entire repo — offer to spawn an Explore agent before doing any file reads
yourself. The agent runs all the reads in parallel without flooding the main
context, and returns a structured inventory you can act on directly.

**When to offer**: any time the target is a directory path (e.g. `src/`) or a
vague scope like "the whole package" or "all the source files".

**Offer text** (adapt as needed):
> "This spans multiple files — I'd recommend spawning an Explore agent to survey
> all `throw`/`@assert`/`error` sites in parallel. It keeps the audit fast and
> leaves the main context clean for the actual rewrites. Want me to do that?"

**Agent prompt to use** (fill in `<path>` and `<package_name>`):

```
Audit `<path>` for error-raising patterns. For every `@assert`, `error(`, or
`throw(` site in every `.jl` file:

1. Record: file, line number, exception type (or "bare @assert" / "bare error"),
   and the full message text (including multiline strings).
2. Classify message quality:
   - "good"  — has `$(expr)` interpolation showing the actual received value, and
     is either short (≤3 lines) or already in a `_throw_` helper function
   - "long-inline" — message content is good, but the body exceeds 3 lines and
     the throw is written inline (not in a `_throw_` helper)
   - "vague" — missing a received value, or no Expected/Got structure
   - "missing" — bare `@assert` with no message at all
3. Note whether the site is at an API boundary (user-facing input) or an internal
   invariant (would require a package bug to fire).

Return a markdown table with columns:
  File | Line | Exception type | Quality | Notes (one-line note on what's wrong if vague/missing/long-inline)

Focus only on sites that are "vague", "missing", or "long-inline" — skip "good" ones.
```

**How to use the result**: treat the returned table as your working inventory for
Steps 1 and 2. You do not need to re-read the flagged files yourself to classify —
go straight to reading only the lines that need rewrites (Step 3 onwards).

---

### Step 1 — Audit the target scope

Identify which files or functions to address. If the user named a specific
function, start there. If the request is repo-wide, run:

```
rg -n '(@assert[^(]|@assert\(|error\(string\(|throw\(ErrorException)' src/
```

Then collect all message-less `@assert` calls:

```
rg -n '@assert' src/ | grep -v '"'
```

Also flag `@warn` calls that use string concatenation instead of interpolation:

```
rg -n '@warn\s+string\(' src/
```

And flag `catch` blocks that discard the original exception when re-throwing:

```
rg -n 'catch\s' src/
```

For each `catch` hit, check whether the subsequent `throw` or `error` call
interpolates the caught variable (e.g. `$e` or `sprint(showerror, e)`). If it
does not, the original exception type and message are silently lost.

For each hit, record: file, line, the condition being checked, and whether it
guards user-provided input (API boundary) or an internal invariant.

### Step 2 — Classify each site

Use this table to choose the right exception type:

| Condition | Exception |
|---|---|
| Invalid user-provided argument | `ArgumentError` |
| Array/matrix shape mismatch | `DimensionMismatch` |
| Inconsistent argument types across parameters | `ArgumentError` |
| Mathematically invalid value (negative variance, etc.) | `DomainError` |
| Invalid index | `BoundsError` |
| Internal invariant that should never fire | `error(...)` |
| Missing interface implementation | `MethodError` or structured `ArgumentError` |

Avoid `ErrorException` unless there is no better choice.

**Type mismatches vs dimension mismatches**: a check that the user supplied
*consistent* arguments (e.g. a feature sampler whose distribution contains the
required "xi" parameter) is checking argument consistency, not size matching.
Use `ArgumentError`, not `DimensionMismatch`, for this pattern.

Distinguish **API boundary** sites (where the user passed something wrong — prefer
typed exceptions with actionable messages) from **internal invariant** sites
(where a bug in the package itself would have to exist — bare `error(...)` with a
clear note is fine there).

**Loop-body errors**: if the throw is inside a `for` or `while` loop, treat the
loop index and key per-iteration state as required context. Without this, the user
sees "matrix is not positive definite" with no idea whether it happened on
training batch 2 or batch 200. Always capture `i` (or the loop variable) and the
state that changed between iterations — the batch index, the feature matrix being
decomposed, etc. For *nested* loops, include both the outer and inner loop
variables. See the loop-context example in the Canonical examples section below.

**`catch e` losing the original exception**: when a Julia exception is caught and
a new one is thrown, the new message must include the original exception. If it
does not, the user loses the root cause (e.g. `PosDefException`, `SingularException`)
and has no way to distinguish a code bug from a numerical issue. Use
`sprint(showerror, e)` rather than `$e` alone — it formats the exception type and
message together:

```julia
# anti-pattern — root cause vanishes
catch e
    throw(ArgumentError("Matrix factorization failed."))
end

# correct — root cause preserved
catch e
    throw(ArgumentError("""
Matrix factorization failed.

Caused by: $(sprint(showerror, e))

Suggestion:
    ...
"""))
end
```

Only suppress the original exception if it is a well-known internal Julia error
(e.g. `SingularException`) and you are intentionally providing a higher-level
fallback — and even then, log it at `@debug` level.

**`@warn string(...)` concatenation and broken interpolation**: two anti-patterns:
`@warn string("...", x, "...")` and `@warn "... cond(regularization) ..."` (a
template string that never interpolates the actual value). Both are equivalent —
the user sees a static message with no actual numbers. Rewrite as
`@warn "... $(cond(regularization)) ..."`. Interpolated `@warn` messages are also
easier to grep and suppress selectively.

**Double-gated invariants**: if a helper is only ever called after the public API
has already checked the same condition, the check inside the helper is an internal
invariant even though it looks like a user-data check. Use a single-line `error(...)`
rather than a full structured `ArgumentError`:

```julia
# internal invariant — the caller already validated this
haskey(params, "xi") || error(
    "Internal error: feature builder called without 'xi' distribution (caller should have validated)",
)
```

### Step 2.5 — Decide: inline or helper?

Before writing the rewrite, decide whether the error belongs inline or should be
extracted into a `_throw_<what>(...)` helper function.

**When to extract** — pull the error into a helper when either condition holds:

- **Length** (primary trigger): the message body exceeds 3 lines. Extract
  unconditionally — single call site, non-loop context, no surrounding complexity
  required. A full Expected / Got / Suggestion block always crosses this threshold.
  Even a one-off long block left inline establishes a pattern that makes entire files
  hard to scan, and accumulates quickly once a few exceptions are made.
- **Duplication**: the same error shape (same summary line, same Expected / Got /
  Suggestion skeleton) appears at ≥2 call sites. Extract even when each block is
  short — the wording drifts silently over time and the call sites collapse to
  readable one-liners.

Inline is appropriate only for genuinely short messages (≤3 lines) at a single
call site. A single summary line, or a summary plus one Got line, is the ceiling
for inline. When in doubt, count — if it doesn't fit in 3 lines, extract.

**Where helpers go**

Default: a `## Error helpers` section at the **bottom of the source file**, above
`end # module`. Keeping helpers near their callers preserves traceability — the
reader sees the throw site, jumps to the bottom of the same file, and finds the
message without switching files.

Promote to a shared `src/ErrorMessages.jl` (or the repo's equivalent top-level
utility file) only when **≥2 different source files** call the same helper. Discover
which file to use by reading the top-level module file (e.g. `src/PackageName.jl`)
for its `include(...)` list — then add `include("ErrorMessages.jl")` as the first
`include` so every subsequent file sees the helpers without any `using`/`import`.

**Submodule scope trap**: When a shared helper lives in `src/ErrorMessages.jl`,
include it *inside every submodule that uses it*, not just in the parent module.
A function defined via `include("ErrorMessages.jl")` in `module RandomFeatures`
is invisible to `module Features` (a nested submodule). The fix: place
`include("ErrorMessages.jl")` at the top of the submodule's own file (e.g.
`Features.jl`), before the files that call it. The package will still load cleanly
with the helper in the wrong scope — only the `@test_throws` tests reveal the
mistake, making this easy to miss in review.

**Naming convention**

```
_throw_<what>(positional_required_facts...; kwargs_for_optional_context...)
```

- Underscore prefix → unexported private helper.
- Verb prefix `_throw_` → the function unconditionally raises; callers know there
  is no return value.
- Suffix describes the failure mode: `_dim_mismatch`, `_missing_xi`,
  `_bad_method`, `_not_iterable`.

**Signature convention**

Pass the facts that are *always* present as positional arguments (the offending
value, the expected vs got summary). Pass *optional* context as keyword arguments
with `nothing` defaults — especially loop context (`index`, `total`, `batch`).
Build optional sections inside the helper by checking `isnothing(...)`.
This keeps call sites compact and lets the same helper serve both loop and non-loop
contexts (see the *Helper with optional loop context* canonical example).

**Performance: use `@noinline`**

Prefix every helper with `@noinline`. This prevents Julia from inlining the cold
error path into the surrounding hot code, keeping numerical kernels unaffected:

```julia
@noinline function _throw_missing_xi(pd; where::Symbol)
    throw(ArgumentError(...))
end
```

**What NOT to do**

- Don't create a catch-all `_throw_arg_error(msg::String)` — that just shifts the
  inline triple-quoted block to another file without any DRY benefit.
- Don't use macros (`@check_xi(...)`) — they're magical and harder to debug than
  plain functions.
- Don't bundle all context into one opaque `context::NamedTuple` — explicit kwargs
  are clearer to call and easier to extend.

### Step 3 — Rewrite with the canonical layout

Use this structure for every user-facing exception:

```julia
throw(ArgumentError("""
Short one-line summary of the failure.

Expected:
    <what would have been valid>

Got:
    <what was actually received, with interpolated values>

Loop context:
    iteration  = $iter (of $n_iter)
    <key per-iteration state variable> = $(summary_of_state)

Context:
    <surrounding state that helps locate the problem>

Suggestion:
    <most likely fix>
"""))
```

Section rules:
- **Summary**: always present; one line; imperative or declarative.
- **Expected / Got**: strongly preferred for any mismatch check; use `$(expr)`
  interpolation to show actual values.
- **Loop context**: include whenever the throw is inside a `for` or `while` loop.
  Always report the loop index and the key state that varies between iterations
  (e.g., the batch number, the feature matrix being decomposed). This is what lets
  the user reproduce the failure without adding `println` debugging.
- **Context**: include when the same error can arise from multiple call sites and
  naming the calling function or struct helps the user orient.
- **Suggestion**: include whenever a likely fix exists. Omit rather than write a
  generic platitude.
- Never dump full matrices or large arrays. Prefer `size(x)`, `eltype(x)`,
  `typeof(x)`, or a scalar summary statistic.

### Step 4 — Move validation early

If the current code lets an invalid input reach a numerical routine before
failing (e.g., `cholesky` on a non-positive-definite matrix, regularization that
is poorly conditioned), add an explicit guard at the API boundary:

```julia
# Before: error surfaces deep in cholesky during predict
cov_chol = cholesky(C)

# After: check at the boundary, raise immediately
isposdef(C) || throw(ArgumentError("""
Feature covariance matrix must be positive definite.

Got:
    size(C)          = $(size(C))
    isposdef(C)      = false
    minimum eigval   ≈ $(minimum(eigvals(Symmetric(C))))

Suggestion:
    Increase the regularization parameter in RandomFeatureMethod.
"""))
cov_chol = cholesky(C)
```

Use `||` for single-condition guards. For multi-condition guards, use `if/throw`.

When using `||` with a multiline triple-quoted throw, the closing `))` goes on its
own line immediately after the closing `"""`:

```julia
condition || throw(ArgumentError("""
Summary line.

Expected:
    ...

Got:
    ...
"""))   # ← closing )) on the line right after the closing """
```

This is the only layout that keeps indentation correct — triple-quoted strings in
Julia do not strip leading whitespace, so indenting the message body would include
those spaces in the string.

### Step 5 — Preserve domain language

Write messages in terms the user understands, not in terms of internal Julia
dispatch or linear algebra internals. For example:

- Say "number of random features" not "size(x, 1)"
- Say "regularization matrix" not "the second argument to cholesky"
- Say "feature parameter distribution" not "the ParameterDistribution object"

### Step 6 — Apply rewrites

Edit each site, keeping the surrounding code untouched. Confirm the package
still loads:

```
julia --project -e 'using RandomFeatures'
```

### Step 7 — Add @test_throws tests

Before writing any test, check whether coverage already exists. Grep the matching
`test/<module>/runtests.jl` for the public API function that reaches the rewritten
site:

```bash
grep -n '@test_throws' test/<module>/runtests.jl | grep '<function_name>'
```

Three outcomes:

| Situation | Action |
|---|---|
| `@test_throws <correct_type>` already present | Skip — do not add a duplicate |
| `@test_throws <wrong_type>` already present | Update the existing line to the new type |
| No coverage at all | Add a new test |

**Check message content, not just type**: In Julia 1.8+, `@test_throws` returns
the caught exception, so you can pin key diagnostic text in the same block:

```julia
let thrown = @test_throws ArgumentError f(bad_input)
    @test contains(thrown.value.msg, "available names")   # Got section present
    @test contains(thrown.value.msg, repr(bad_input))     # received value interpolated
end
```

Add at least one `contains` check per new error site. Without this, a refactor can
preserve the exception type while silently dropping the Got section or the
interpolated value, and no test will catch it. Check for: a phrase from the summary
line, the Got section label (e.g. `"available names"`, `"missing required keys"`),
and `repr(bad_value)` when the message uses `repr`. The `let` block is the cleanest
form — it keeps the type assertion and the content assertions co-located.

For every site that needs a new test, add it in the matching `test/<module>/runtests.jl`:

```julia
@test_throws ArgumentError ScalarFeature(100, bad_sampler, Cosine())
```

Use the specific exception type — never bare `@test_throws Exception`. The test
should construct the minimal invalid input that triggers the new error, without
duplicating happy-path coverage.

**Update existing tests that used the wrong type.** If the file already has a
`@test_throws ErrorException` (or any other type) for a site you're rewriting to
`ArgumentError`, update that existing test in the same edit.

**Testing unexported helpers.** If the site is inside an unexported helper,
do not `import` the internal directly. Instead, test through the nearest exported
public API function that calls it, using invalid input that propagates to the helper:

```julia
# The "xi" check is inside ScalarFeature's constructor — test via the exported constructor
pd_without_xi = constrained_gaussian("theta", 0.0, 1.0)  # no "xi" component
@test_throws ArgumentError ScalarFeature(100, FeatureSampler(pd_without_xi), Cosine())
```

This keeps tests coupled to the public contract and avoids brittleness when
internal function names change.

### Step 8 — Offer to improve the skill

Once the rewrites and tests are clean, offer: "Would you like to improve the
**error-message-manager** skill itself using skill-creator? You can share
suggestions, or I can analyse patterns from this session—recurring edge cases,
exception-type decisions, or anything that felt awkward—to refine the skill for
next time."

---

## Style rules

- **Triple-quoted strings** for all multiline messages.
- **No full matrix dumps**. Use `size(x)`, `eltype(x)`, `norm(x - ...)`, or
  `extrema(x)` instead.
- **Interpolate actual values** in Got sections so the user sees the numbers,
  not just variable names. For `String`-typed arguments use `$(repr(x))` rather
  than `$(x)` — it adds the surrounding quotes so the output clearly reads as a
  string value (e.g. `Got: method = "bad"` instead of `Got: method = bad`).
- **Raise early**: prefer guarding at the function entry point over deep inside a
  helper.
- **No `@assert` for user-facing validation**. `@assert` is a debugging tool;
  it can be compiled out. Use explicit `throw` instead.
- **Loop state in Got / Loop context**: when a throw is inside a `for` or `while`
  loop, always name the iteration index and the key state from that iteration.
  A "factorization failed" message without the batch index forces the user to add
  `println` debugging to reproduce the failure. When the same loop-context check
  recurs at multiple sites, the loop variables become optional kwargs on a
  `_throw_<what>` helper.
- **Preserve the original exception in `catch` blocks**: if you catch `e` and
  throw a new exception, include `$(sprint(showerror, e))` in the new message.
  Dropping `e` silently discards the root cause.
- **`@warn` with interpolation, not `string()`**: replace
  `@warn string("κ(reg) = cond(regularization)")` with
  `@warn "κ(reg) = $(cond(regularization))"`. Uninterpolated template strings
  in warnings show only the literal variable name, not its value.
- **Single-line messages are fine** when the failure is unambiguous and no
  Expected/Got context would add clarity.
- **Extract into `_throw_<what>(...)` helpers** whenever the message body exceeds
  3 lines, or when the same Expected / Got / Suggestion skeleton appears at ≥2
  call sites (even if short). A full Expected / Got / Suggestion block always
  exceeds 3 lines and must be a helper — inline is only appropriate for ≤3-line
  messages at a single call site. Place the helper in a `## Error helpers` section
  at the bottom of the source file; promote to a shared `src/ErrorMessages.jl` only
  when ≥2 different source files share the helper. Use `@noinline`, positional args
  for required facts, and `nothing`-defaulted kwargs for optional context such as
  loop indices. Render each optional section only when its kwarg is non-`nothing`.

---

## Canonical before/after examples

> **Length rule applies to all examples below.** Each example shows the canonical
> message *format* (Expected / Got / Suggestion sections, interpolation, etc.). When
> the message body exceeds 3 lines — which any structured block with Expected / Got /
> Suggestion sections does — the throw must go in a `_throw_<what>(...)` helper per
> Step 2.5, not inline. The first example below models this explicitly. Subsequent
> examples show the message body format; apply the same helper extraction whenever
> the resulting message exceeds 3 lines.

### Replace a vague ArgumentError (missing received values)

The after-message has 10 lines (above the 3-line threshold), so it goes into a
`_throw_` helper — extract unconditionally at this length even though there is
only one call site.

```julia
# Before — no interpolation: user sees no names from their distribution
if "xi" ∉ get_name(get_parameter_distribution(feature_sampler))
    throw(
        ArgumentError(
            " Named parameter \"xi\" not found in names of parameter_distribution. " *
            " \n Please provide the name \"xi\" to the distribution used to sample the features",
        ),
    )
end

# After — helper in the ## Error helpers section at the bottom of the file
@noinline function _throw_missing_xi(pd; where::Symbol)
    throw(ArgumentError("""
$where: parameter distribution must include a component named "xi" for feature sampling.

Expected:
    "xi" ∈ get_name(parameter_distribution)

Got:
    available names = $(get_name(pd))

Suggestion:
    Add a ParameterDistribution component named "xi" when constructing FeatureSampler,
    e.g. via constrained_gaussian("xi", μ, σ).
"""))
end

# Call site collapses to a single guard line:
"xi" ∈ get_name(get_parameter_distribution(feature_sampler)) ||
    _throw_missing_xi(get_parameter_distribution(feature_sampler); where = :ScalarFeature)
```

### Replace a bare `@assert` on an API boundary

```julia
# Before
@assert(haskey(batch_sizes, "train") && haskey(batch_sizes, "test") && haskey(batch_sizes, "feature"))

# After
all(k -> haskey(batch_sizes, k), ("train", "test", "feature")) || throw(ArgumentError("""
batch_sizes dict is missing required keys.

Expected keys:
    "train", "test", "feature"

Got keys:
    $(collect(keys(batch_sizes)))

Suggestion:
    Provide all three keys: Dict("train" => 0, "test" => 0, "feature" => 0).
"""))
```

### Replace a string-value error with `repr`

```julia
# Before — user sees  method = svdd  with no quotes, hard to see it's a string typo
throw(ArgumentError(
    "Only factorization methods \"pinv\", \"cholesky\" and \"svd\" implemented. got " * string(method)
))

# After — repr keeps quotes visible, making string typos obvious
throw(ArgumentError("""
Unrecognised matrix factorisation method.

Expected:
    "pinv", "cholesky", or "svd"

Got:
    method = $(repr(method))
"""))
```

Using `repr(method)` rather than `$(method)` keeps the string quotes visible in
the output, making it unambiguous that the user passed a `String` value (and
making copy-paste typos easy to spot).

### Replace a dimension-mismatch guard (or add one that is missing)

```julia
# Before — no check; size mismatch surfaces deep inside a linear algebra call
mean_store = zeros(output_dim, n_test)   # pre-allocated buffer

# After — explicit guard before the computation
size(mean_store) == (output_dim, size(inputs, 2)) || throw(DimensionMismatch("""
Pre-allocated mean_store has the wrong shape for this prediction.

Expected:
    size(mean_store) == (output_dim, n_test)

Got:
    size(mean_store) = $(size(mean_store))
    output_dim       = $output_dim
    n_test           = $(size(inputs, 2))
"""))
```

### Preserve the original exception when catching and re-throwing

```julia
# Before — PosDefException or SingularException silently discarded
try
    cov_chol = cholesky(feature_cov)
catch e
    error("Feature covariance factorization failed.")
end

# After — root cause preserved, matrix state shown
try
    cov_chol = cholesky(feature_cov)
catch e
    throw(ArgumentError("""
Feature covariance factorization failed during prediction.

Got:
    size(feature_cov)    = $(size(feature_cov))
    isposdef(feature_cov) = $(isposdef(feature_cov))

Caused by: $(sprint(showerror, e))

Suggestion:
    The regularization parameter in RandomFeatureMethod may be too small.
    Try increasing it so the feature covariance stays positive definite.
"""))
end
```

`sprint(showerror, e)` formats as `"LinearAlgebra.PosDefException: matrix is not
Hermitian; Cholesky factorization failed."` — far more informative than `string(e)`.

### Rewrite broken `@warn` interpolation

```julia
# Before — condition number never actually interpolated (template string, not interpolation)
@warn "The provided regularization is poorly conditioned: κ(reg) = cond(regularization). Imprecision or SingularException during inversion may occur."

# After — actual value shown
κ = cond(regularization)
@warn "Regularization matrix is poorly conditioned (κ = $κ, threshold = 1e8). Inversion may lose precision or throw SingularException."

# Before (with string concatenation)
@warn string("Regularization is not positive definite.", "\n Applying posdef correction.")

# After
@warn "Regularization matrix is not positive definite — applying posdef correction."
```

### Add loop context to an error thrown inside a batch loop

```julia
# Before — user sees "not positive definite" with no idea which batch failed
for i in 1:n_batches
    try
        cov_chol = cholesky(C_batch[i])
    catch e
        error("Cholesky factorization failed")
    end
end

# After — guard before cholesky, expose batch index and diagnostic state
for i in 1:n_batches
    isposdef(C_batch[i]) || throw(ArgumentError("""
Feature covariance is not positive definite at training batch $i / $n_batches.

Expected:
    A positive-definite covariance matrix at every batch step.

Got:
    batch index        = $i / $n_batches
    size(C_batch[i])   = $(size(C_batch[i]))
    minimum eigval     ≈ $(minimum(eigvals(Symmetric(C_batch[i]))))

Suggestion:
    Ensemble collapse or under-regularisation can cause this near batch $i.
    Consider increasing the regularization parameter in RandomFeatureMethod.
"""))
    cov_chol = cholesky(C_batch[i])
end
```

Key points:
- **Move the guard before the failing call** so the message fires with full batch
  state still in scope.
- **Report the loop variable** (`i`) and its upper bound so the user knows whether
  the failure is early (batch 2/200) or late (batch 198/200).
- **Include one diagnostic scalar** — the minimum eigenvalue rather than dumping
  the full matrix.

### Extract a duplicated error into a helper

The "xi" parameter guard appears identically in both `ScalarFeature` and
`VectorFeature` constructors. Before extraction, byte-for-byte identical 8-line
blocks appear at both call sites:

```julia
# Before — same block in both ScalarFeature and VectorFeature constructors

# in ScalarFeature:
if "xi" ∉ get_name(get_parameter_distribution(feature_sampler))
    throw(ArgumentError(
        " Named parameter \"xi\" not found in names of parameter_distribution. " *
        " \n Please provide the name \"xi\" to the distribution used to sample the features",
    ))
end

# in VectorFeature: byte-for-byte identical block
if "xi" ∉ get_name(get_parameter_distribution(feature_sampler))
    throw(ArgumentError(
        " Named parameter \"xi\" not found in names of parameter_distribution. " *
        " \n Please provide the name \"xi\" to the distribution used to sample the features",
    ))
end
```

After extraction, both constructors call one helper defined in `src/ErrorMessages.jl`
(promoted there because two different source files share it):

```julia
# After — shared helper in src/ErrorMessages.jl

## Error helpers

@noinline function _throw_missing_xi(pd; where::Symbol)
    throw(ArgumentError("""
$where: parameter distribution must include a component named "xi" for feature sampling.

Expected:
    "xi" ∈ get_name(parameter_distribution)

Got:
    available names = $(get_name(pd))

Suggestion:
    Add a ParameterDistribution component named "xi" when constructing FeatureSampler,
    e.g. via constrained_gaussian("xi", μ, σ).
"""))
end

# Both call sites collapse to a single readable guard line each:
# in ScalarFeature constructor:
"xi" ∈ get_name(get_parameter_distribution(feature_sampler)) ||
    _throw_missing_xi(get_parameter_distribution(feature_sampler); where = :ScalarFeature)

# in VectorFeature constructor:
"xi" ∈ get_name(get_parameter_distribution(feature_sampler)) ||
    _throw_missing_xi(get_parameter_distribution(feature_sampler); where = :VectorFeature)
```

Key points:
- The `where::Symbol` kwarg embeds the calling type name so diagnostics stay
  specific even though the body is shared.
- `@noinline` keeps the error path out of the hot constructor body.
- The helper lives in `src/ErrorMessages.jl` because two source files call it —
  the rule for promoting out of a single file.

### Helper with optional loop context

The feature-parameter validation loop currently reports no position when a dict
is missing a required key — the user sees "missing sigma" with no idea which entry
in the array triggered it. Extracting into a helper adds the index and makes the
same helper reusable wherever that validation appears:

```julia
# Before — inline block, no loop index in the message
for params in feature_param_list
    if "sigma" ∉ keys(params)
        throw(ArgumentError("""
Feature parameter dict is missing required key "sigma".

Expected keys:
    "sigma" (required), plus any additional feature-specific parameters

Got keys:
    $(sort(collect(string.(keys(params)))))

Suggestion:
    Ensure each parameter dict includes "sigma" => <value>.
"""))
    end
end

# After — helper with optional loop context at the bottom of the file

@noinline function _throw_missing_sigma(got_keys; index = nothing, total = nothing)
    loop_ctx = isnothing(index) ? "" : """

Loop context:
    param index = $index (of $total)"""
    throw(ArgumentError("""
Feature parameter dict is missing required key "sigma".$loop_ctx

Expected keys:
    "sigma" (required), plus any additional feature-specific parameters

Got keys:
    $got_keys

Suggestion:
    Ensure each parameter dict includes "sigma" => <value>.
"""))
end

# Call site — loop now reports position:
for (i, params) in enumerate(feature_param_list)
    "sigma" ∈ keys(params) ||
        _throw_missing_sigma(
            sort(collect(string.(keys(params)))); index = i, total = length(feature_param_list),
        )
end

# The same helper works outside a loop — omit the kwargs and the Loop context
# section is silently suppressed:
"sigma" ∈ keys(feature_parameters) ||
    _throw_missing_sigma(sort(collect(string.(keys(feature_parameters)))))
```

Key points:
- `index` and `total` default to `nothing`; the `Loop context:` section is
  rendered only when they are provided.
- Switching `for params in ...` to `for (i, params) in enumerate(...)` is the
  only loop-side change needed to expose the index.
- The user now knows *which* parameter dict failed, not just that one of them did.

---

## Non-goals

- Do not rewrite every low-level exception in the package. Focus on user-facing
  API boundaries and sites explicitly identified.
- Do not suppress Julia stack traces. The goal is clearer diagnostics, not
  silenced errors.
- Do not add verbosity for its own sake. A short, clear message beats a long,
  generic one.
- Do not expose internal linear algebra variable names or dispatch details when
  domain-level terminology exists.
- Do not extract truly short errors (≤3 lines) at a single call site — the
  inline form is easier to grep and keeps cause and message co-located. A single
  summary line, or a summary plus one Got line, is the ceiling for inline.
