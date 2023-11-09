# reference in tree version of RandomFeatures
prepend!(LOAD_PATH, [joinpath(@__DIR__, "..")])

using Documenter,
    RandomFeatures, RandomFeatures.Samplers, RandomFeatures.Features, RandomFeatures.Methods, RandomFeatures.Utilities

# Gotta set this environment variable when using the GR run-time on CI machines.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

api = [
    "Samplers" => "API/Samplers.md",
    "Features" => "API/Features.md",
    "Methods" => "API/Methods.md",
    "Utilities" => "API/Utilities.md",
]

pages = [
    "Home" => "index.md",
    "Installation instructions" => "installation_instructions.md",
    "Scalar method" => "setting_up_scalar.md",
    "Vector method" => "setting_up_vector.md",
    "Bottlenecks and performance tips" => "parallelism.md",
    "Contributing" => "contributing.md",
    "API" => api,
]



format = Documenter.HTML(collapselevel = 1, prettyurls = !isempty(get(ENV, "CI", "")))

makedocs(
    sitename = "RandomFeatures.jl",
    authors = "CliMA Contributors",
    format = format,
    pages = pages,
    modules = [RandomFeatures],
    doctest = true,
    clean = true,
    checkdocs = :none,
)

if !isempty(get(ENV, "CI", ""))
    deploydocs(
        repo = "github.com/CliMA/RandomFeatures.jl.git",
        versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
        push_preview = true,
        devbranch = "main",
    )
end
