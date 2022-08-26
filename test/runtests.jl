using Test


TEST_PLOT_FLAG = !isempty(get(ENV, "TEST_PLOT_FLAG", ""))

if TEST_PLOT_FLAG
    using Plots, ColorSchemes
end


function include_test(_module)
    println("Starting tests for $_module")
    t = @elapsed include(joinpath(_module, "runtests.jl"))
    println("Completed tests for $_module, $(round(Int, t)) seconds elapsed")
    return nothing
end

@testset "EnsembleKalmanProcesses" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false

    function has_submodule(sm)
        any(ARGS) do a
            a == sm && return true
            first(split(a, '/')) == sm && return true
            return false
        end
    end

    for submodule in [
        "Utilities",
        "Samplers",
        "Features",
        "Methods",
    ]
        if all_tests || has_submodule(submodule) || "RandomFeatures" in ARGS
            include_test(submodule)
        end
    end
end
