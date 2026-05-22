## Error helpers shared across multiple source files

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
