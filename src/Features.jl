module Features

include("ScalarFunctions.jl")

import StatsBase: sample
import RandomFeatures.Samplers: get_optimizable_parameters

using EnsembleKalmanProcesses.ParameterDistributions, DocStringExtensions, RandomFeatures.Samplers

export RandomFeature, ScalarFeature, ScalarFourierFeature, ScalarNeuronFeature

export sample,
    get_optimizable_parameters,
    get_scalar_function,
    get_feature_sampler,
    get_feature_sample,
    get_n_features,
    get_hyper_sampler,
    get_hyper_fixed,
    build_features

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
    hyper_sampler::Union{Sampler, Nothing}
    hyper_fixed::Union{Dict, Nothing}
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
    hyper_sampler::Union{Sampler, Nothing} = nothing,
    hyper_fixed::Union{Dict, Nothing} = nothing,
)
    if "xi" ∉ get_name(get_parameter_distribution(feature_sampler))
        throw(
            ArgumentError(
                " named parameter \"xi\" not found in names of parameter_distribution. " *
                " \n Please provide the name \"xi\" to the distribution used to sample the features",
            ),
        )
    end

    # TODO: improve this horrible check
    no_hyper = 0
    if isnothing(hyper_fixed)
        no_hyper += 1
    elseif "sigma" ∉ keys(hyper_fixed)
        no_hyper += 1
    end
    if no_hyper == 1
        if isnothing(hyper_sampler)
            no_hyper += 1
        elseif "sigma" ∉ get_name(get_parameter_distribution(hyper_sampler))
            no_hyper += 1
        end
    end
    if no_hyper == 2
        throw(
            ArgumentError(
                "No value for multiplicative feature scaling \"sigma\" set." *
                "\n If \"sigma\" is to be fixed:" *
                "\n Construct a ScalarFeature with keyword hyper_fixed = Dict(\"sigma\" => value)," *
                "\n If \"sigma\" is to be learnt from data:" *
                "\n Create a Sampler for a distribution named \"sigma\", then construct a ScalarFeature with keyword hyper_sampler=...",
            ),
        )
    end

    hf = hyper_fixed
    if !isnothing(hyper_fixed)
        if "sigma" ∈ keys(hyper_fixed)
            if !isnothing(hyper_sampler)
                if "sigma" ∈ get_name(get_parameter_distribution(hyper_sampler))
                    @info "both a `hyper_fixed=` and `hyper_sampler=` specify \"sigma\"," *
                          "\n defaulting to optimize \"sigma\" with hyper_sampler"
                    hf = nothing # remove the unused option
                end
            end
        end
    end


    samp = sample(feature_sampler, n_features)

    return ScalarFeature(n_features, feature_sampler, scalar_fun, samp, hyper_sampler, hf)
end

#these call the above constructor
"""
$(TYPEDSIGNATURES)

Constructor for a `Sampler` with cosine features
"""
function ScalarFourierFeature(
    n_features::Int,
    sampler::Sampler;
    hyper_sampler::Union{Sampler, Nothing} = nothing,
    hyper_fixed = nothing,
)
    return ScalarFeature(n_features, sampler, Cosine(), hyper_sampler = hyper_sampler, hyper_fixed = hyper_fixed)
end

"""
$(TYPEDSIGNATURES)

Constructor for a `Sampler` with activation-function features
"""
function ScalarNeuronFeature(
    n_features::Int,
    sampler::Sampler;
    activation_fun::ScalarActivation = Relu(),
    hyper_sampler::Union{Sampler, Nothing} = nothing,
    hyper_fixed = nothing,
)
    return ScalarFeature(n_features, sampler, activation_fun, hyper_sampler = hyper_sampler, hyper_fixed = hyper_fixed)
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

get_hyper_sampler(rf::RandomFeature) = rf.hyper_sampler
get_hyper_fixed(rf::RandomFeature) = rf.hyper_fixed

function get_optimizable_parameters(rf::RandomFeature)

end

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
    features = sqrt(2) * apply_scalar_function(sf, features)

    is_sigma_scaling = "sigma" ∈ get_name(samp)
    if is_sigma_scaling
        sigma = get_distribution(samp)["sigma"]
    else
        sigma = get_hyper_fixed(rf)["sigma"] # scalar
    end
    features *= sigma

    return features # n
end

build_features(rf::ScalarFeature, inputs::AbstractMatrix) = build_features(rf, inputs, collect(1:get_n_features(rf)))


end #module
