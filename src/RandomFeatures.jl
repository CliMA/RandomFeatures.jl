module RandomFeatures

using
    Statistics,
    LinearAlgebra,
    DocStringExtensions

#auxiliary modules
include("Utilities.jl") # some additional tools
include("Samplers.jl") # samples a distribution
include("Features.jl") # builds a feature from the samples
include("Methods.jl") # fits to data
#include("HyperparameterOptimizers.jl") # optimizes hyperparameters

end # module
