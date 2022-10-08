module Features

include("ScalarFunctions.jl")

import StatsBase: sample

using EnsembleKalmanProcesses.ParameterDistributions, DocStringExtensions, RandomFeatures.Samplers

export RandomFeature, ScalarFeature, ScalarFourierFeature, ScalarNeuronFeature

export sample,
    get_scalar_function, get_feature_sampler, get_feature_sample, get_n_features, build_features, get_feature_parameters

abstract type RandomFeature end

"""
$(TYPEDSIGNATURES)

samples the random feature distribution 
"""
function sample(rf::RandomFeature)
    sampler = get_feature_sampler(rf)
    m = get_n_features(rf)
    return sample(sampler, m)
end


"""
$(TYPEDEF)

Contains information to build and sample RandomFeatures mapping from N-D -> 1-D

$(TYPEDFIELDS)
"""
struct ScalarFeature <: RandomFeature
    "Number of features"
    n_features::Int
    "Sampler of the feature distribution"
    feature_sampler::Sampler
    "ScalarFunction mapping R -> R"
    scalar_function::ScalarFunction
    "Current `Sample` from sampler"
    feature_sample::ParameterDistribution
    "hyperparameters in Feature (and not in Sampler)"
    feature_parameters::Union{Dict, Nothing}
end

# common constructors
"""
$(TYPEDSIGNATURES)

basic constructor for a `ScalarFeature'
"""
function ScalarFeature(
    n_features::Int,
    feature_sampler::Sampler,
    scalar_fun::ScalarFunction;
    feature_parameters::Dict = Dict("sigma" => 1),
)
    if "xi" ∉ get_name(get_parameter_distribution(feature_sampler))
        throw(
            ArgumentError(
                " Named parameter \"xi\" not found in names of parameter_distribution. " *
                " \n Please provide the name \"xi\" to the distribution used to sample the features",
            ),
        )
    end

    if "sigma" ∉ keys(feature_parameters)
        @info(" Required feature parameter key \"sigma\" not defined, continuing with default value \"sigma\" = 1 ")
        feature_parameters["sigma"] = 1.0
    end

    samp = sample(feature_sampler, n_features)

    return ScalarFeature(n_features, feature_sampler, scalar_fun, samp, feature_parameters)
end

#these call the above constructor
"""
$(TYPEDSIGNATURES)

Constructor for a `Sampler` with cosine features
"""
function ScalarFourierFeature(n_features::Int, sampler::Sampler; feature_parameters::Dict = Dict("sigma" => sqrt(2.0)))
    return ScalarFeature(n_features, sampler, Cosine(); feature_parameters = feature_parameters)
end

"""
$(TYPEDSIGNATURES)

Constructor for a `Sampler` with activation-function features
"""
function ScalarNeuronFeature(n_features::Int, sampler::Sampler; activation_fun::ScalarActivation = Relu(), kwargs...)
    return ScalarFeature(n_features, sampler, activation_fun; kwargs...)
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
get_scalar_function(rf::ScalarFeature) = rf.scalar_function

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


"""
$(TYPEDSIGNATURES)

builds features (possibly batched) from an input matrix of size (input dimension,number of samples)
"""
function build_features(
    rf::ScalarFeature,
    inputs_t::AbstractMatrix, # input_dim x n_sample
    batch_feature_idx::AbstractVector{Int},
)
    inputs = permutedims(inputs_t, (2, 1)) # n_sample x input_dim

    # build: sigma * sqrt(2) * scalar_function(xi . input + b)
    samp = get_feature_sample(rf)
    xi = get_distribution(samp)["xi"] # dim_inputs x n_features
    features = inputs * xi[:, batch_feature_idx] # n_samples x n_features

    is_uniform_shift = "uniform" ∈ get_name(samp)
    if is_uniform_shift
        uniform = get_distribution(samp)["uniform"] # 1 x n_features
        features .+= uniform[:, batch_feature_idx]
    end

    sf = get_scalar_function(rf)
    features = apply_scalar_function(sf, features)

    sigma = get_feature_parameters(rf)["sigma"] # scalar
    features *= sigma

    return features # n
end

build_features(rf::ScalarFeature, inputs::AbstractMatrix) = build_features(rf, inputs, collect(1:get_n_features(rf)))


end #module
