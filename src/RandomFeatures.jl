module RandomFeatures

using
    Statistics,
    LinearAlgebra,
    DocStringExtensions

#auxiliary modules
include("Samplers.jl") # samples a distribution
include("Features.jl") # builds a feature from the samples
#include("RandomFeatureMethod.jl") # fits to data
#include("HyperparameterOptimizers.jl") # optimizes hyperparameters

end # module
