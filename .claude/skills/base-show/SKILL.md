---
name: base-show
description: >
  Add concise Base.show and Base.summary methods to Julia types whose default REPL
  representation is unhelpful or overwhelming. Use this skill whenever the user
  mentions that a type prints badly in the REPL, asks to improve how an object is
  displayed or printed, wants a custom show, summary, or repr for a Julia type, or
  says the REPL output is noisy, verbose, or hard to read. Also trigger when the user
  asks to "make the REPL output nicer", "add a show method", "add a summary method",
  "customize display", or "fix what prints when I type a variable name". This skill
  produces compact, informative Base.show and Base.summary methods and matching unit
  tests — invoke it proactively whenever show, summary, display, print, repr,
  or REPL output is mentioned in a Julia context. Good candidates in RandomFeatures.jl
  include Decomposition, ScalarFeature, VectorFeature, RandomFeatureMethod, and Fit —
  all of which hold matrices or nested objects that print poorly by default.
---

# base-show

Add concise `Base.show(io::IO, ::MIME"text/plain", x::T)` and `Base.summary(io::IO,
x::T)` methods to Julia types whose default REPL representation is unhelpful or
overwhelming. Julia's default show dumps every field recursively; types that hold
matrix decompositions, large parameter distributions, nested arrays, or many scalar
fields produce screens of unreadable text at the REPL.

`Base.show(io, MIME"text/plain", x)` must also handle the `:compact` IOContext key.
When Julia renders an object as an element inside a container (e.g. printing a
`Vector{MyType}`), it sets `:compact => true` on `io`. Without a compact branch the
full multi-line output is repeated for every element, producing an unreadable wall of
text. The compact branch must produce exactly one line (no newlines), giving the same
kind of at-a-glance hint as `Base.summary`.

This skill produces both methods and accompanying unit tests so that interactive use
of the package is pleasant without losing key summary information.

## Workflow

### Step 0 — Audit existing show methods (retrofit mode)

Skip this step if you are adding show methods to types that have none. Apply it when
the user asks to retrofit existing show methods — e.g. to add the compact branch to
methods that were written before this protocol existed.

**Find MIME methods that lack the compact branch:**

```
grep -n 'MIME"text/plain"' src/show.jl
```

For each match, check whether the function body contains `get(io, :compact`. Any that
do not are candidates for retrofit.

**Detect the old forwarding anti-pattern (infinite-recursion risk):**

```
grep -nA2 'function Base\.show(io::IO, x::' src/ | grep 'show(io, MIME'
```

If this matches, a 2-arg `show(io, x)` is calling the MIME method — the *wrong*
direction. Once the MIME method gains a compact branch that calls `show(io, x)`, you
get infinite recursion. Flag every match and reverse the direction: the 2-arg method
becomes the compact one-liner, and the MIME method calls it via `show(io, x)` in its
compact branch.

**Identify pre-existing bespoke 2-arg shows:**

A bespoke 2-arg show is one that already exists but does not follow summary style —
for example, it may omit the type name entirely or use a different format. Check each
existing `Base.show(io::IO, x::T)` against its paired `Base.summary`. If the outputs
differ substantially, the 2-arg show is bespoke and needs a custom compact test (see
Step 4).

### Step 1 — Enumerate concrete types

List every concrete (non-abstract) struct defined in the package source:

```
grep -nrE '^(mutable )?struct ' src/
```

Exclude `abstract type` declarations — they cannot be instantiated and do not need
show methods.

### Step 2 — Classify show noisiness

For each concrete type, decide whether its default show output would be noisy. A type
is noisy if it holds at least one of:

- A matrix decomposition or large `AbstractMatrix`
- A `ParameterDistribution` or similar composite distribution object
- A `Dict` with potentially many entries
- A large or variable-length `Array`
- Another struct that is itself noisy
- More than approximately six fields in total

Also run:

```
grep -nrE 'Base\.(show|summary)' src/
```

Skip any type that already has a custom `Base.show` or `Base.summary` method — do not
overwrite existing customization.

Also skip **zero-field structs** — their default show (`Cosine()`, `Relu()`) is already
compact and informative; there is nothing to improve.

### Step 3 — Write show and summary methods

For each noisy type without existing methods, write **both** a `Base.show` and a
`Base.summary` method.

**`Base.show`** — always write two overloads together:

```julia
# 3-arg MIME method: full REPL display, with compact fallback
function Base.show(io::IO, ::MIME"text/plain", x::T)
    if get(io, :compact, false)
        show(io, x)   # delegate to the 2-arg compact method
    else
        println(io, "T")
        println(io, "  field_name : ", summary_value)
        # ...
    end
end

# 2-arg method: single-line compact representation (no newline)
function Base.show(io::IO, x::T)
    print(io, "T (key_hint)")
end
```

The 3-arg (MIME) non-compact branch must:

- Print the type name (and any cheap size hints) on the first line.
- Follow with 1–5 concise summary lines: counts, sizes, or ranges of important fields.
  Never print collection contents.
- Produce at most 10 lines of output for any valid instance, including edge cases such
  as empty collections or zero-element structs.

The 2-arg method (compact representation) must:

- Produce exactly one line with no trailing newline.
- Match `Base.summary` style: type name followed by the most essential identifying hint
  in parentheses — e.g. `"ScalarFeature (256 features, Cosine)"`.
- Remain O(1): no loops, no collection materialisation.

Julia calls the 2-arg method when rendering elements inside containers (arrays, dicts,
etc.), passing `io` with `:compact => true`. The MIME method's compact branch delegates
to it so both paths produce the same single-line output.

**`Base.summary`** — single-line description used when the object appears inside a
container or is printed in a broader context (e.g., as an element of a `Vector`):

```julia
function Base.summary(io::IO, x::T)
    print(io, "T (key_hint)")
end
```

The method must:

- Fit on one line — no newlines.
- Convey the most important size or identity hint (e.g., number of features, output
  dimension, matrix size), so the reader immediately knows what they are looking at.
- Remain cheap: O(1) field accesses only.

Good examples of what to put in the hint: `"256 features"`, `"10×10 SVD"`,
`"empty"`. Avoid repeating the type name verbatim as the only content — add value.

**Placement**: place both methods adjacent to their type definition in the same source
file, or gather all show/summary methods in a dedicated `src/show.jl` included from
the main module file. Follow whatever convention is already present in the package;
default to `src/show.jl` if no prior convention exists.

If creating `src/show.jl`, add `include("show.jl")` to the main module file after the
type definitions it references.

**Multi-submodule packages**: When each `.jl` file opens its own `module` (e.g.
`module Utilities ... end`), the types live in separate namespaces. In this case,
include `show.jl` from the *parent* module file **after all submodule includes**, and
reference types by their submodule-qualified name:

```julia
# In show.jl, included from module RandomFeatures after all submodule includes:
function Base.show(io::IO, ::MIME"text/plain", x::Utilities.Decomposition) ...
function Base.show(io::IO, x::Features.ScalarFeature) ...
```

Inside show methods placed at the parent-module level, **prefer direct field access**
(`x.n_features`) over calling submodule accessors (`Features.get_n_features(x)`) —
the accessor may not be in scope, and field access is just as clear.

### Step 4 — Write unit tests

Write one test block per type, covering `show` (full and compact), and `summary`. Each
test block must:

- Construct a minimal valid instance of the type.
- For full show: capture output with `sprint(show, MIME("text/plain"), instance)` and
  assert that it contains the type name and that line count does not exceed 10.
- For compact show: capture `out2 = sprint(show, instance)` (2-arg) and assert it
  contains the type name and has no `'\n'`. Also capture
  `out3 = sprint(show, MIME("text/plain"), instance; context=:compact => true)` and
  assert `out2 == out3` — both compact paths must agree.
- For `summary`: capture output with `sprint(summary, instance)` and assert that it
  contains the type name and produces exactly one line (no `'\n'` in output).

**Test runner registration**: If the package has a top-level `test/runtests.jl` that
iterates over a list of submodule test directories, add `"show"` to that list so the
new tests run as part of `Pkg.test()`.

**DRY helpers for multiple types**: When testing more than two types, extract the
repeating assertions into helpers to avoid noisy repetition:

```julia
function check_full(x, typename)
    out = sprint(show, MIME("text/plain"), x)
    @test occursin(typename, out)
    @test count(==('\n'), out) <= 10
end
function check_compact(x, typename)
    out2 = sprint(show, x)
    @test occursin(typename, out2) && !occursin('\n', out2)
    @test out2 == sprint(show, MIME("text/plain"), x; context = :compact => true)
end
function check_summary(x, typename)
    out = sprint(summary, x)
    @test occursin(typename, out) && !occursin('\n', out)
end
```

Then each type's test block becomes three lines: `check_full(x, "T")`,
`check_compact(x, "T")`, `check_summary(x, "T")`. Add any type-specific assertions
(e.g., that the feature count appears in the compact hint) after the helpers.

**Bespoke 2-arg shows (retrofit case):** Some types may already have a 2-arg show
that intentionally does not include the type name or follow summary style — the method
is doing something custom. Using a shared `check_compact(x, typename)` helper will
fail the typename assertion for these. Instead, write a hand-rolled compact test:

```julia
s2 = sprint(show, instance)
@test !occursin('\n', s2)                                              # no newline
@test s2 == sprint(show, MIME("text/plain"), instance; context = :compact => true)  # paths agree
```

Avoid asserting exact strings so that cosmetic changes to the output do not break tests.

### Step 5 — Verify

Run the package test suite:

```
julia --project -e 'using Pkg; Pkg.test()'
```

Confirm that all new tests pass and no pre-existing tests regress.

### Step 6 — Offer to improve the skill

After the tests pass and the REPL output looks good, ask the user: "Would you like to improve the **base-show** skill itself using skill-creator? You can suggest changes to the workflow or quality criteria, or I can analyse what came up during this session to identify improvements to the skill."

## Common patterns

### Two-overload pattern (always write both together)

Always define the 2-arg and 3-arg MIME overloads as a pair. The MIME method's compact
branch calls the 2-arg method, so both display paths (REPL and in-container) converge
on the same one-liner without repetition:

```julia
function Base.show(io::IO, ::MIME"text/plain", x::RandomFeatureMethod)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "RandomFeatureMethod")
        # ... full multi-line body ...
    end
end

function Base.show(io::IO, x::RandomFeatureMethod)
    print(io, "RandomFeatureMethod (", nameof(typeof(x.random_feature)),
          ", n=", get_n_features(x.random_feature), ")")
end
```

Without the 2-arg method, `[rfm]` in a `Vector` falls back to Julia's default field
dump. Without the compact branch in the MIME method, the same dump appears whenever
the object is embedded in a container that happens to call `show(io, MIME"text/plain",
x)` with `:compact => true`.

### Truncate long collections with "… and N more"

When a type holds a variable-length collection, cap the loop to keep output bounded:

```julia
function Base.show(io::IO, ::MIME"text/plain", x::Fit)
    n = length(x.coeffs)
    println(io, "Fit with ", n, " coefficient", n == 1 ? "" : "s")
    max_show = 8
    for i in 1:min(n, max_show)
        println(io, "  [", i, "]: ", round(x.coeffs[i], sigdigits = 4))
    end
    n > max_show && println(io, "  … and ", n - max_show, " more")
end
```

### Conditional fields

Only print a field when it carries information:

```julia
if !isnothing(x.feature_parameters)
    println(io, "  feature_params : ", sprint(summary, x.feature_parameters))
end
```

### Pluralisation in summary

Match English grammar for counts that can be 0 or 1:

```julia
print(io, "ScalarFeature (", n, " feature", n == 1 ? "" : "s", ", ", sf_name, ")")
```

### Arrow notation for mappings

Use `→` in summary when the type represents a transformation between spaces:

```julia
print(io, "VectorFeature (", n_features, " features → ", output_dim, "D output)")
```

### Unicode in mathematical contexts

Use `×` for matrix dimensions, `→` for transformations, and `|u|` for set sizes.
These are rendered cleanly in all modern Julia terminals and communicate mathematical
meaning concisely.

```julia
# Decomposition summary: Decomposition (10×10, SVD)
m, n = size(x.full_matrix)
print(io, "Decomposition (", m, "×", n, ", ", nameof(typeof(x.decomposition)), ")")
```

### Use nameof for parametric type identity

When a type carries a type-parameter that identifies its variant, use `nameof` rather
than printing the full parameterised name:

```julia
# ScalarFeature{Cosine} → print "Cosine", not the full type string
print(io, "ScalarFeature (", x.n_features, " features, ",
      nameof(typeof(x.scalar_function)), ")")
```

### Section separators in show.jl

When collecting all methods in a dedicated `show.jl`, organise by type family with
aligned comment rulers:

```julia
# ── ScalarFeatures ────────────────────────────────────────────────────────────
# ── VectorFeatures ────────────────────────────────────────────────────────────
# ── RandomFeatureMethod / Fit ─────────────────────────────────────────────────
# ── Utilities / Decomposition ─────────────────────────────────────────────────
```

## Quality criteria

| Criterion | Priority | Definition |
|---|---|---|
| Coverage | High | Every type classified as noisy in Step 2 has a `Base.show` (both overloads) and a `Base.summary` method. |
| Compact support | High | The 3-arg MIME `show` checks `get(io, :compact, false)` and calls the 2-arg `show(io, x)` in the compact branch. The 2-arg method produces exactly one line with no newline. |
| Brevity — show | High | Full (non-compact) show output is at most 10 lines for any valid instance, including edge cases. |
| Brevity — summary | High | Summary output is exactly one line (no newlines) for any valid instance. |
| Safety | High | Neither method throws on any valid instance. |
| Allocation-safety | High | All data access is O(1): use `length()`, `size()`, `isempty()`, or `first()` on lazy iterators. Never call `collect()`, `sort()`, `filter()`, or any function that materialises a new collection. |
| Test robustness | Medium | Tests assert structural properties, not exact strings. Cosmetic changes do not break tests. |
| No regression | High | Pre-existing tests continue to pass; no unintended changes to other source files. |

## Formatting rules

- **MIME show signature**: `Base.show(io::IO, ::MIME"text/plain", x::MyType)`
- **MIME show structure**: always starts with `if get(io, :compact, false); show(io, x); else ... end`.
- **MIME show full branch — first line**: type name via `println(io, "TypeName")`. Cheap size hints may follow on the same line.
- **MIME show full branch — subsequent lines**: indented two spaces for readability.
- **2-arg show signature**: `Base.show(io::IO, x::MyType)`
- **2-arg show content**: one `print` call (no `println`), type name followed by a parenthesised hint matching `Base.summary` style, e.g. `print(io, "Decomposition (10×10)")`.
- **summary signature**: `Base.summary(io::IO, x::MyType)`
- **summary content**: one `print` call (no `println`), type name followed by a parenthesised hint, e.g. `print(io, "ScalarFeature (256 features, Cosine)")`.
- **No collection contents**: print only counts, sizes, or ranges — never iterate and print elements.
- **No allocations**: use `length()`, `size()`, `isempty()`, and `first()` on lazy iterators. Do not call `collect()`, `sort()`, or any function that copies a collection.
- **Tests — MIME full show**: use `sprint(show, MIME("text/plain"), x)` to capture output without side effects.
- **Tests — compact show**: use `sprint(show, MIME("text/plain"), x; context=:compact => true)` to exercise the compact branch, and `sprint(show, x)` to test the 2-arg method directly.
- **Tests — summary**: use `sprint(summary, x)` to capture the one-line description.

## Examples

### Example 1 — matrix-carrying type (size hint)

```julia
# Scenario: a type wraps three matrices — the original, its factorisation, and its
# inverse — used to solve linear systems efficiently during prediction.

# Before (default Julia show — prints all three matrices in full)
julia> dc
Decomposition(full_matrix=[10×10 Float64 Matrix], decomposition=SVD{Float64,...}(...),
  inv_decomposition=[10×10 Float64 Matrix])

# After — custom show (two overloads)
function Base.show(io::IO, ::MIME"text/plain", x::Decomposition)
    if get(io, :compact, false)
        show(io, x)
    else
        m, n = size(x.full_matrix)
        println(io, "Decomposition")
        println(io, "  size  : ", m, " × ", n)
        println(io, "  method: ", nameof(typeof(x.decomposition)))
    end
end

function Base.show(io::IO, x::Decomposition)
    m, n = size(x.full_matrix)
    print(io, "Decomposition (", m, "×", n, ")")
end

# julia> dc
# Decomposition
#   size  : 10 × 10
#   method: SVD

# julia> [dc, dc]
# 2-element Vector{Decomposition}:
#  Decomposition (10×10)
#  Decomposition (10×10)

# After — custom summary (matches 2-arg show)
function Base.summary(io::IO, x::Decomposition)
    m, n = size(x.full_matrix)
    print(io, "Decomposition (", m, "×", n, ")")
end
```

### Example 2 — composite type with nested objects

```julia
# Scenario: a type bundles a feature count, an activation function, and a large
# ParameterDistribution of sampled feature parameters — printing it by default
# dumps the entire distribution in full.

# Before (default Julia show — nested ParameterDistribution printed in full)
julia> sf
ScalarFeature{Cosine}(n_features=256, feature_sampler=Sampler{MersenneTwister}(
  parameter_distribution=ParameterDistribution{...}(...), rng=MersenneTwister(...)),
  scalar_function=Cosine(), feature_sample=ParameterDistribution{...}(...),
  feature_parameters=Dict("sigma"=>1))

# After — custom show (two overloads)
function Base.show(io::IO, ::MIME"text/plain", x::ScalarFeature)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "ScalarFeature")
        println(io, "  n_features     : ", x.n_features)
        println(io, "  scalar_function: ", nameof(typeof(x.scalar_function)))
        if !isnothing(x.feature_parameters)
            println(io, "  feature_params : ", sprint(summary, x.feature_parameters))
        end
    end
end

function Base.show(io::IO, x::ScalarFeature)
    print(io, "ScalarFeature (", x.n_features, " feature",
          x.n_features == 1 ? "" : "s", ", ", nameof(typeof(x.scalar_function)), ")")
end

# julia> sf
# ScalarFeature
#   n_features     : 256
#   scalar_function: Cosine
#   feature_params : Dict{String, Any} with 1 entry

# julia> [sf, sf]
# 2-element Vector{ScalarFeature{Cosine}}:
#  ScalarFeature (256 features, Cosine)
#  ScalarFeature (256 features, Cosine)

# After — custom summary (matches 2-arg show)
function Base.summary(io::IO, x::ScalarFeature)
    print(io, "ScalarFeature (", x.n_features, " feature",
          x.n_features == 1 ? "" : "s", ", ", nameof(typeof(x.scalar_function)), ")")
end
```

### Example 3 — configuration type with regularisation and batch sizes

```julia
# Scenario: a configuration struct holds a random feature object, a regularisation
# matrix (which may be large), and a batch-size dict. Printing it by default dumps
# all nested objects including the full regularisation matrix.

# Before (default Julia show — dumps all nested objects)
julia> rfm
RandomFeatureMethod{ScalarFeature{Cosine}}(random_feature=ScalarFeature{Cosine}(...),
  batch_sizes=Dict("train"=>0, "test"=>0, "feature"=>0),
  regularization=UniformScaling{Float64}(2.220446049250313e-10),
  tullio_threading=true)

# After — custom show (two overloads)
function Base.show(io::IO, ::MIME"text/plain", x::RandomFeatureMethod)
    if get(io, :compact, false)
        show(io, x)
    else
        bs = x.batch_sizes
        println(io, "RandomFeatureMethod")
        println(io, "  feature type  : ", nameof(typeof(x.random_feature)))
        println(io, "  n_features    : ", get_n_features(x.random_feature))
        println(io, "  regularization: ", nameof(typeof(x.regularization)))
        println(io, "  batch_sizes   : train=", bs["train"],
                    ", test=", bs["test"], ", feature=", bs["feature"])
        println(io, "  threading     : ", x.tullio_threading)
    end
end

function Base.show(io::IO, x::RandomFeatureMethod)
    print(io, "RandomFeatureMethod (", nameof(typeof(x.random_feature)),
          ", n=", get_n_features(x.random_feature), ")")
end

# julia> rfm
# RandomFeatureMethod
#   feature type  : ScalarFeature
#   n_features    : 256
#   regularization: UniformScaling
#   batch_sizes   : train=0, test=0, feature=0
#   threading     : true

# julia> [rfm, rfm]
# 2-element Vector{RandomFeatureMethod{...}}:
#  RandomFeatureMethod (ScalarFeature, n=256)
#  RandomFeatureMethod (ScalarFeature, n=256)

# After — custom summary (matches 2-arg show)
function Base.summary(io::IO, x::RandomFeatureMethod)
    print(io, "RandomFeatureMethod (", nameof(typeof(x.random_feature)),
          ", n=", get_n_features(x.random_feature), ")")
end
```

### Unit tests

```julia
@testset "Decomposition show" begin
    dc = Decomposition(rand(8, 8), "svd")
    out = sprint(show, MIME("text/plain"), dc)
    @test occursin("Decomposition", out)
    @test count(==('\n'), out) <= 10
end

@testset "Decomposition show compact" begin
    dc = Decomposition(rand(8, 8), "svd")
    # exercise via the 2-arg method directly
    out2 = sprint(show, dc)
    @test occursin("Decomposition", out2)
    @test !occursin('\n', out2)
    # exercise via the MIME method with compact context
    out3 = sprint(show, MIME("text/plain"), dc; context = :compact => true)
    @test out2 == out3   # both paths must agree
end

@testset "Decomposition summary" begin
    dc = Decomposition(rand(8, 8), "svd")
    out = sprint(summary, dc)
    @test occursin("Decomposition", out)
    @test !occursin('\n', out)    # must be exactly one line
end
```
