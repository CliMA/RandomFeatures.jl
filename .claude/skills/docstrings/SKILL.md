---
name: docstrings
description: >
  Add or normalise Julia docstrings on public symbols (exported types, functions,
  and constants) so the package's public API is fully self-documenting and the
  Documenter.jl docs build passes its checkdocs check. After writing docstrings,
  also updates docs/src/API/ pages so every exported symbol appears exactly once,
  organised into logical categories, with stale entries removed.
  Invoke this skill whenever the user mentions: docstring, missing doc,
  undocumented symbol, API doc, checkdocs warning, docs/src/API, @docs block,
  or asks to document a type or function, sync the API pages, or keep the API
  index up to date. Also use it when the user asks to "write docs for" or "add
  docs to" source files, or when a CI failure mentions missing or incomplete
  docstrings. In RandomFeatures.jl, the 2026-05-20 audit resolved all bare stubs;
  common failure modes to watch for are inaccurate return-type claims, getter prose
  copy-pasted across dispatches, and jldoctests using `using PackageName` for symbols
  that live only in a submodule and need `using PackageName.SubModule`.
---

# docstrings

Add or normalise Julia docstrings on public symbols (exported types, functions,
and constants) across the package source. The goal is complete, consistent API
documentation that renders correctly under Documenter.jl and follows whichever
docstring convention is already established in the package — in RandomFeatures.jl
this is the DocStringExtensions new format: `$(TYPEDEF)` + prose + `$(TYPEDFIELDS)`
for structs, and `$(TYPEDSIGNATURES)` + prose for functions. Completing this skill
makes the package's public API fully self-documenting and satisfies any `checkdocs`
requirement in the docs build.

## Workflow

### Step 1 — Detect the existing convention

Use an Explore subagent to read 2–3 symbols that already have complete docstrings
to calibrate style. This avoids consuming the main context window with large file
reads. Ask the Explore agent to return the verbatim docstring text for each symbol.

Identify:

- Whether DocStringExtensions macros are used, and which ones (`$(TYPEDEF)`,
  `$(TYPEDFIELDS)`, `$(TYPEDSIGNATURES)`, `$(METHODLIST)`).
- How prose is structured relative to macro-generated content (e.g. does prose
  come before or after `$(TYPEDFIELDS)`?).
- What field documentation pattern is preferred: inline string literals above
  each struct field vs. a separate prose block.
- Which format is used for struct docstrings: the **old format** (an indented
  type-name header on the first line, no `$(TYPEDEF)`, manual `# Constructor`
  section), or the **new format** (prose only, `$(TYPEDEF)` for the signature,
  `$(METHODLIST)` for constructors). Normalise old-format structs to new-format
  during Step 3.

This detected baseline becomes the style target for every new or normalised
docstring. Do not impose a different convention — match what is already there.

### Step 2 — Enumerate candidates

Discover the package name from `Project.toml` (the `name =` field). Then run:

```
grep -nE '^(function |struct |abstract type |mutable struct |const )' src/**/*.jl
```

**Cross-file exports**: exported names may be declared in a central module file
(e.g. `src/PackageName.jl`) while the definition lives in a different file.
Read the module file for all `export` statements so you catch every public symbol
regardless of where it is defined.

For each exported symbol, check whether a non-empty docstring immediately precedes
the definition. Produce a prioritised list:

1. **Missing entirely** — no docstring at all.
2. **Old-format struct** — indented type name on first line, no `$(TYPEDEF)`,
   or redundant manual `# Constructor` / `# Constructors` section alongside
   `$(METHODLIST)`.
3. **Empty or stub** — only a bare macro line (e.g. `$(TYPEDSIGNATURES)`) with
   no prose.
4. **Incomplete** — prose present but key sections absent (missing `# Arguments`,
   `# Examples`, or field strings not describing semantic role).

**Accuracy checks on existing prose.** Alongside the coverage gaps above, run two
quick passes so correctness bugs in *existing* docstrings don't survive the edit cycle:

- **Return-type claims.** For any function whose existing docstring prose contains
  "Returns a " or "Returns an ", read the actual `return` statement(s) in the source
  and verify the claimed shape matches. A common failure mode is a one-liner wrapper
  whose docstring says "Returns an `output_dim × n_samples` array" but the
  implementation returns a `(array, features)` tuple. Flag these as category 4
  (Incomplete) and correct the prose in Step 3.

- **Getter semantic claims.** For getter functions (names beginning with `get_`)
  that assert a specific value or type in prose — e.g. "equals 1", "returns a Bool",
  "always positive" — verify the claim against the actual implementation body for
  *every* dispatch of that getter name. Copy-paste errors across overloads are common:
  a `get_output_dim` defined on a vector type may have inherited prose that says
  "equals 1 for scalar-valued features". Flag these as category 4 and correct the
  prose so each overload's docstring only makes claims that hold for its own dispatch.

**Also scan every function in the file for old-style docstrings**, regardless of
whether it is exported. An old-style docstring is one that uses an indented
function-name header (e.g. `    my_func(arg1, arg2)`) and/or an `Args:` /
`Arguments:` block with the `` `name` - description `` format. Convert these to
the `$(TYPEDSIGNATURES)` convention in the same editing pass — the whole file
should end up stylistically uniform.

### Step 3 — Draft docstrings

For each candidate, write a docstring that matches the detected convention.

#### Getters and simple accessors

Simple getters for exported types (e.g. `get_n_features(rf::RandomFeature)`,
`get_rng(s::Sampler)`) are public API and must be documented if exported. A
one-line `$(TYPEDSIGNATURES)` + short prose sentence suffices; no `# Arguments`
or `# Examples` is needed unless the semantics are non-obvious.

#### Old-format struct docstrings

If a struct docstring starts with an indented type name (e.g. `    MyStruct{...}`),
convert it to the new format:

- Remove the indented type name from the docstring.
- Add `$(TYPEDEF)` immediately after the opening prose sentence.
- Replace any manual `# Constructor` or `# Constructors` section (listing
  function signatures) with a `# Constructors` section containing only
  `$(METHODLIST)`. If `$(METHODLIST)` is already present alongside the manual
  list, remove the manual list.
- Preserve any genuine prose that was in the old `# Constructor` section if it
  explains non-obvious behaviour; discard boilerplate signature repetition.
- After adding `$(METHODLIST)`, check whether any exported factory function
  builds this struct under a different name (e.g. `fit` for `Fit`,
  `FeatureSampler` for `Sampler`). If so, insert a prose note naming that
  factory in the `# Constructors` section, immediately before `$(METHODLIST)`,
  following the **Named constructors** pattern below. Do *not* bury the factory
  reference only in the opening prose — putting it inside `# Constructors` is
  what makes it discoverable to readers scanning for "how do I build one of
  these?".

#### Named constructors (factory functions)

`$(METHODLIST)` only lists methods whose name matches the struct type. Exported
functions that **build an instance of the struct but carry a different name** —
factory functions such as `FeatureSampler` for `Sampler`, or `ScalarFourierFeature`
and `ScalarNeuronFeature` for `ScalarFeature` — are invisible to `$(METHODLIST)`
and must be surfaced manually in the struct docstring.

When such functions exist, add a prose note inside the `# Constructors` section,
immediately before `$(METHODLIST)`:

```julia
"""
Wrap the parameter distributions used to sample random features.

$(TYPEDEF)

# Fields

$(TYPEDFIELDS)

# Constructors

Recommended construction is via the `FeatureSampler()` utility, which assembles
the sampler from a feature distribution and an optional bias distribution.
Convenience factories `ScalarFourierFeature` and `ScalarNeuronFeature` build the
full `ScalarFeature` (including its sampler) in a single call.

$(METHODLIST)
"""
struct Sampler{RNG <: AbstractRNG}
    ...
end
```

The note should:
- Name the function in backticks.
- Give a one-line hint about when to prefer it over the direct constructor.
- Optionally include a minimal usage snippet if the factory is the primary entry
  point and no separate `# Examples` block exists on the factory function itself.

The factory function still needs its **own full docstring** (`$(TYPEDSIGNATURES)`,
`# Arguments`, `# Examples` if non-trivial). The struct-level note is a pointer,
not a replacement.

**Detecting named constructors during Step 2:** when enumerating candidates, flag
exported functions whose name differs from any type name but whose body or doc
clearly returns an instance of a specific struct. Common signals: the function name
ends with a domain term (e.g. `ScalarFourierFeature`, `FeatureSampler`), its return
statement calls the struct constructor directly, or the existing codebase already
mentions the relationship somewhere in prose.

#### Multiple dispatch — one docstring per concept

When a function has multiple dispatch methods, document **only** the primary
user-facing overload and leave all other overloads undocumented. Competing
docstrings fragment the rendered API docs and create maintenance burden.

The primary overload is the method whose argument type is the **broadest
user-facing type** — e.g. `RandomFeature` rather than `ScalarFeature` or
`VectorFeature`.

**Type-parameter specialisations count as overloads.** If the same function name
is defined for multiple concrete feature types, these are dispatch methods of the
same concept. Document only one — typically the first defined, or the more general
one — and leave the rest undocumented.

Internal framework methods that are exported for extensibility but not called
directly by end users do not need documentation even if exported.

**Overloads with different return types are separate concepts.** The one-docstring
rule is for type-parameter specialisations that do the same thing for different
argument types. It does *not* apply when two overloads of the same name return
fundamentally different shapes — e.g. a 3-argument form that returns a
`(result, features)` tuple while a 4-argument form returns only `result`. These
are distinct contracts and both deserve a docstring, each with a clear description
of what it returns.

Exception: if both variants are only ever called through a single, higher-level
public function (e.g. `predict` wraps both `predictive_mean` and `predictive_cov`
internally), leave the implementation variants undocumented and document only the
public entry point. The test is whether a user would ever plausibly call the variant
directly; if not, it does not need a docstring.

#### Old-style function docstrings (all functions, not just exported)

Convert any docstring that uses an indented function-name header or an `Args:` /
`Arguments:` section to the `$(TYPEDSIGNATURES)` style, even for internal
(non-exported) helpers. The canonical old-style markers are:

- First line indented with spaces: `    my_func(arg, ...)` — replace with `$(TYPEDSIGNATURES)`.
- Argument block labelled `Args:` or `Arguments:` with `` `name` - description `` lines — replace
  with a `# Arguments` section using `` - `name`: description `` format.

Doing this in the same pass keeps the file stylistically uniform and prevents
old-style docstrings from persisting as invisible technical debt.

#### General rules

- Use the same macro set as the best-documented symbols already in the package.
- Preserve any inline field string literals already present above struct fields —
  do not merge them into the struct-level docstring.
- Prose should answer: what does this symbol represent or do, when would a caller
  use it, and what are the physical units of key quantities.
- Do not duplicate content that macros generate automatically (e.g. do not
  restate field types when `$(TYPEDFIELDS)` already renders them).
- Physical quantities: always include units in square brackets, e.g. `[m/day]`.
- For functions with more than two arguments, or whose argument semantics are
  not obvious from the name alone, add a `# Arguments` section listing each
  parameter as `` - `name`: description [unit if applicable] ``.
- For every non-trivial public function where a minimal runnable example can be
  written, add a `# Examples` section with a `jldoctest` block so Documenter.jl
  can verify the example stays correct as the code evolves.

### Step 4 — Apply edits

When editing files that contain non-ASCII characters (e.g. author names with
accented letters), the file may store characters in Unicode NFD form while the
Edit tool normalises to NFC, causing match failures. If an Edit call fails with
a "not found" error on a string you can see in the file, use a Python one-liner
to apply the replacement with NFD-normalised strings:

```bash
python3 - <<'EOF'
import unicodedata, pathlib
p = pathlib.Path("src/MyFile.jl")
text = p.read_text()
old = unicodedata.normalize('NFD', "the old string here")
new = unicodedata.normalize('NFD', "the new string here")
p.write_text(text.replace(old, new, 1))
EOF
```

After a Python edit, re-read the file before making any further Edit calls to
the same file (the Edit tool tracks file state from the last Read).

### Step 5 — Sync `docs/src/API/` pages

After all source-file edits are applied, update the Documenter.jl API pages so
that every exported, documented symbol appears exactly once, organised into
logical categories. The goal is that a reader browsing `docs/src/API/` sees a
complete, non-redundant index of the public API — nothing missing, nothing
stale.

#### 5a — Build the source-to-page map

Read `docs/make.jl` and extract the `api` array to see which display name maps
to which page path (e.g. `"Methods" => "API/Methods.md"`). For each page,
read its `@meta` block to find `CurrentModule = ...`. This tells you which
module's exports the page is responsible for.

#### 5b — Collect current `@docs` entries per page

For each API page, extract every symbol entry listed inside ` ```@docs ``` `
blocks. Some entries carry type-signature qualifiers (e.g.
`get_n_features(rf::ScalarFeature)`) — track both the raw entry string and
the base name (everything before the first `(`).

#### 5c — Find missing and stale entries

**Exported but not defined (phantom exports).** Before anything else, check
that every exported name actually resolves to a definition — a function, type,
or constant — somewhere in the source files of that module. If an exported name
has no definition anywhere, it is a phantom export: remove the `export`
statement (or just that name from a multi-name `export` line) from the source
file. Do not add phantom exports to any API page.

**Missing from the API.** A symbol is **missing** from a page when it is
exported from that page's `CurrentModule`, its base name does not appear in any
`@docs` block on any API page, and it has a definition in the source. If it
lacks a docstring, go back and write one now (following the conventions from
Steps 1–3) before adding it to the API page — an undocumented entry in a
`@docs` block will cause the docs build to error. Every exported, defined
symbol must end up with a docstring and an API page entry.

**Stale API entries.** An entry is **stale** when the base name is no longer
exported from the module, or the symbol no longer has a definition in the
source.

Run all three checks before making edits so you can see the full diff in one
pass.

#### 5d — Place missing symbols into appropriate sections

Insert each missing symbol into the section of its API page that best matches
its role. Use the existing section headings on the page as the primary guide —
`## Getter functions`, `## Error metrics`, etc. are already established
categories; add the new symbol to the most thematically fitting one.

When no existing section fits, create a new `##` heading that names the
functional group (e.g. `## Factory functions`, `## Utility functions`) and open
a fresh ` ```@docs ``` ` block below it. Avoid catch-all sections like
`## Miscellaneous`; if you find yourself reaching for that, split more finely.

Broad heuristics for categorisation when the page has no existing sections to
guide you:

- Struct / abstract type → primary types section (first block on page)
- Functions starting with `get_` → `## Getter functions`
- Functions starting with `compute_`, `construct_`, `build_` → a computation
  or construction section
- Factory functions (e.g. `ScalarFourierFeature`, `FeatureSampler`) → `## Factory functions`
- Update or step functions → an operations section
- Error-metric functions → `## Error metrics`

For a multiple-dispatch function where only the primary overload is documented
(per Step 3), list only that overload. If the existing page convention uses
type-qualified entries (e.g. `get_n_features(rf::RandomFeature)`), follow that
convention; otherwise use the plain name.

#### 5e — Remove stale entries

For each stale API entry:

1. Delete the line from its `@docs` block. If that empties the block, delete
   the block. If that empties the section, delete the section heading too.
2. If the symbol is stale because it is no longer defined (phantom export),
   also remove the `export` statement from the source file. For multi-name
   export lines (e.g. `export foo, bar, baz`), remove only the stale name and
   leave the rest intact.

#### 5f — Ensure no symbol appears on two pages

Each base name must appear on at most one API page. If you find a duplicate,
keep it on the page whose `CurrentModule` matches the module where the symbol
is defined, and remove it from the other page.

### Step 6 — Verify

Find the package name from `Project.toml`, then confirm the package loads
without error:

```
julia --project -e 'import Pkg; Pkg.instantiate(); using RandomFeatures'
```

If a docs build is configured (`docs/make.jl` is present), run it and resolve
any `checkdocs` warnings introduced by the new docstrings.

### Step 7 — Offer to improve the skill

Once the docs build is clean, ask the user: "Would you like to improve the
**docstrings** skill itself using skill-creator? You can share suggestions, or I
can analyse patterns from this session — recurring edge cases, formatting
decisions, or anything that felt awkward — to refine the skill for next time."

## Formatting rules

These rules encode the conventions most Julia packages following DocStringExtensions
expect. Apply them consistently.

- **Triple-quoted strings** for all docstrings.
- **First line**: concise one-line summary — imperative mood for functions
  (`"Return the..."`, `"Compute..."`), noun phrase for types and constants.
- **Second line**: blank.
- **Body**: prose, then any macro invocations. `$(TYPEDSIGNATURES)` must be the
  very first line of a function docstring and is the sole source of the method
  signature — never write a manual indented signature as well.
- **No trailing whitespace** inside the docstring.
- **No emojis.**
- **Physical units** in square brackets: `[m/day]`, `[kg/m³]`, `[day]`, etc.
- **Field string literals** (the string above each struct field) are distinct
  from the struct-level docstring. Preserve both; do not merge them.
- Field string literals must describe the field's *semantic role*, not its type.
  Never write a type name inside brackets (e.g. `"[Dict]"`) —
  `$(TYPEDFIELDS)` already renders the type. Reserve square-bracket notation
  exclusively for physical units.
- Avoid vague labels such as "data object" or "container". Say what the field
  represents in domain terms (e.g. "number of random features used in the
  approximation" rather than "integer count").
- **Multiple-dispatch — one docstring per concept**: Document only the primary
  user-facing overload (the method taking the top-level composite type). All
  other dispatch methods remain undocumented. Do **not** add `$(METHODLIST)` to
  function docstrings — `$(TYPEDSIGNATURES)` already surfaces all overloads.
  `$(METHODLIST)` belongs only in struct docstrings (inside `# Constructors`).
  **Exception — convenience overloads with distinct kwargs**: If a secondary overload
  introduces keyword arguments or a shorthand interface not visible in the primary
  overload's signature (e.g. an `output_dim::Int` convenience argument or a
  `uniform_shift_bounds` keyword), do not silently delete that information. Instead,
  remove the secondary overload's docstring and fold a brief `# Convenience overloads`
  note into the primary docstring, naming the extra kwargs and when to prefer them.
- **`# Arguments` section**: add after the opening prose for any function with
  more than two parameters, or where argument semantics are non-obvious. Format:
  `` - `name`: description [unit] ``.
- **`# Examples` section**: add for every non-trivial public function where a
  minimal runnable example is feasible. Use `jldoctest` blocks with `julia> `
  prompts and include expected output.
- In every `jldoctest` block, separate each `julia> ` prompt from the next with
  a blank line. Documenter.jl rejects blocks where two prompts appear
  consecutively without an intervening blank line. If a statement produces no
  output, end it with a semicolon and add a blank line before the next prompt.
- **jldoctest import line**: Before writing any `# Examples` block, verify that
  the symbol is actually accessible via `using PackageName` at the top level. Read
  the main module file (`src/PackageName.jl`) and check whether the symbol appears
  in a top-level `export` statement or is brought in via `using .SubModule`. If
  neither is true, the symbol lives only in the submodule — use the full path as
  the import line: `julia> using PackageName.SubModule`. Using the wrong import
  causes a silent doctest failure (`UndefVarError`) even though the symbol is
  genuinely public. Do not assume the package is already in scope.

## Quality criteria

| Criterion | Weight | What to check |
|---|---|---|
| **Completeness** | High | Every exported symbol has a non-empty docstring after the task is applied. |
| **Convention parity** | High | New docstrings use the same macro set and structural pattern as the best-documented symbols already present. Old-format struct docstrings have been normalised. |
| **Informativeness** | Medium | Prose answers "what, when, why". Units present for physical quantities. `# Arguments` section present where needed. `# Examples` jldoctest block present for non-trivial public functions. |
| **No duplication** | Medium | Prose does not duplicate macro-generated content. Field string literals do not restate the field's type. No redundant manual `# Constructor` section alongside `$(METHODLIST)`. |
| **API page coverage** | High | Every exported, documented symbol appears exactly once across `docs/src/API/` pages. No stale entries. Symbols are grouped into descriptive sections. |
| **Correctness** | High | Package loads without error; docs build (if configured) completes without new warnings. |

## Examples

### Struct: old format → new format

```julia
## Before — OLD format: indented type name, no $(TYPEDEF), manual # Constructor section

"""
    Fit{V<:AbstractVector, USorM<:Union{UniformScaling, AbstractMatrix}}

Holds the fitted random feature coefficients and matrix decomposition.

# Constructor
Fit(feature_factors, coeffs, regularization)

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct Fit{V <: AbstractVector, USorM <: Union{UniformScaling, AbstractMatrix}}
    ...
end

## After — NEW format: $(TYPEDEF), prose, $(TYPEDFIELDS), and factory method in # Constructors
## (factory `fit` has a different name from the struct, so $(METHODLIST) adds nothing;
##  the # Constructors section is pure prose naming the factory instead)

"""
$(TYPEDEF)

Holds the coefficients and matrix decomposition produced by `fit`, describing a trained random
feature regression model. Pass to `predict`, `predictive_mean`, or `predictive_cov` to obtain
predictions at new input locations.

$(TYPEDFIELDS)

# Constructors

Produced by `fit(rfm, input_output_pairs; decomposition_type = "cholesky")` — not intended to
be constructed directly.
"""
struct Fit{V <: AbstractVector, USorM <: Union{UniformScaling, AbstractMatrix}}
    ...
end
```

### Function: stub docstring improved

```julia
## Before — function with an empty stub docstring

"""
$(TYPEDSIGNATURES)
"""
function Decomposition(mat::AbstractMatrix, method::String; nugget::Real = 1e12 * eps())
    ...
end

## After — prose, Arguments, and Examples sections added

"""
$(TYPEDSIGNATURES)

Compute and store a matrix factorisation of `mat` using `method`, together with its
inverse, for efficient repeated linear solves.

# Arguments
- `mat`: the square matrix to factorise.
- `method`: factorisation strategy — one of `"svd"`, `"cholesky"`, or `"pinv"`.
- `nugget`: small regularisation constant added to the diagonal before factorising [`Float64`].

# Examples
```jldoctest
julia> using RandomFeatures

julia> M = [4.0 2.0; 2.0 3.0];

julia> dc = Decomposition(M, "cholesky");

julia> size(get_decomposition(dc))
(2, 2)
```
"""
function Decomposition(mat::AbstractMatrix, method::String; nugget::Real = 1e12 * eps())
    ...
end
```

### Multiple-dispatch: type-parameter specialisations

```julia
## Before — both specialisations documented (anti-pattern)

"""
$(TYPEDSIGNATURES)

Return the predictive mean for a scalar random feature model.
"""
function predictive_mean(rfm::RandomFeatureMethod, inputs::AbstractMatrix,
                         fitted::Fit; kwargs...)  # ScalarFeature path
    ...
end

"""
$(TYPEDSIGNATURES)

Return the predictive mean for a vector random feature model.
"""
function predictive_mean(rfm::RandomFeatureMethod, inputs::AbstractMatrix,
                         fitted::Fit; kwargs...)  # VectorFeature path
    ...
end

## After — only the first overload documented; second left bare

"""
$(TYPEDSIGNATURES)

Return the predictive mean of the fitted random feature model evaluated at `inputs`.
"""
function predictive_mean(rfm::RandomFeatureMethod, inputs::AbstractMatrix,
                         fitted::Fit; kwargs...)
    ...
end

function predictive_mean(rfm::RandomFeatureMethod, inputs::AbstractMatrix,
                         fitted::Fit; kwargs...)
    ...
end
```

### Simple getter: minimal docstring

```julia
## Before — getter with no docstring

get_n_features(rf::RandomFeature) = rf.n_features

## After — one-liner is enough

"""
$(TYPEDSIGNATURES)

Return the number of random features in `rf`.
"""
get_n_features(rf::RandomFeature) = rf.n_features
```
