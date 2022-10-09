module Features

include("ScalarFunctions.jl")
import StatsBase: sample

using EnsembleKalmanProcesses.ParameterDistributions, DocStringExtensions, RandomFeatures.Samplers, Tullio

abstract type RandomFeature end
include("ScalarFeatures.jl")
include("VectorFeatures.jl")
export RandomFeature


export sample,
    get_scalar_function, get_feature_sampler, get_feature_sample, get_n_features, get_feature_parameters, get_output_dim


"""
$(TYPEDSIGNATURES)

samples the random feature distribution 
"""
function sample(rf::RandomFeature)
    sampler = get_feature_sampler(rf)
    m = get_n_features(rf)
    return sample(sampler, m)
end

# methods
"""
$(TYPEDSIGNATURES)

gets the `n_features` field 
"""
get_n_features(rf::RandomFeature) = rf.n_features

"""
$(TYPEDSIGNATURES)

gets the `scalar_function` field 
"""
get_scalar_function(rf::RandomFeature) = rf.scalar_function

"""
$(TYPEDSIGNATURES)

gets the `feature_sampler` field 
"""
get_feature_sampler(rf::RandomFeature) = rf.feature_sampler

"""
$(TYPEDSIGNATURES)

gets the `feature_sample` field 
"""
get_feature_sample(rf::RandomFeature) = rf.feature_sample

"""
$(TYPEDSIGNATURES)

gets the `feature_parameters` field 
"""
get_feature_parameters(rf::RandomFeature) = rf.feature_parameters



end #module
