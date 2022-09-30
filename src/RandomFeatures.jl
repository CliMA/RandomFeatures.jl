"""
# Imported modules:
$(IMPORTS)

# Exports:
$(EXPORTS)
"""
module RandomFeatures

using Statistics, LinearAlgebra, DocStringExtensions

# importing parameter distirbutions
import EnsembleKalmanProcesses: ParameterDistributions, DataContainers

export ParameterDistributions, DataContainers

#auxiliary modules
include("Utilities.jl") # some additional tools
include("Samplers.jl") # samples a distribution
include("Features.jl") # builds a feature from the samples
include("Methods.jl") # fits to data

end # module
