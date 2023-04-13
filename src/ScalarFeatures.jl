export ScalarFeature, ScalarFourierFeature, ScalarNeuronFeature
export build_features

"""
$(TYPEDEF)

Contains information to build and sample RandomFeatures mapping from N-D -> 1-D

$(TYPEDFIELDS)
"""
struct ScalarFeature{S <: AbstractString, SF <: ScalarFunction} <: RandomFeature
    "Number of features"
    n_features::Int
    "Sampler of the feature distribution"
    feature_sampler::Sampler
    "ScalarFunction mapping R -> R"
    scalar_function::SF
    "Current `Sample` from sampler"
    feature_sample::ParameterDistribution
    "hyperparameters in Feature (and not in Sampler)"
    feature_parameters::Union{Dict{S}, Nothing}
end

# common constructors
"""
$(TYPEDSIGNATURES)

basic constructor for a `ScalarFeature'
"""
function ScalarFeature(
    n_features::Int,
    feature_sampler::Sampler,
    scalar_fun::SF;
    feature_parameters::Dict{S} = Dict("sigma" => 1),
) where {S <: AbstractString, SF <: ScalarFunction}
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

    return ScalarFeature{S, SF}(n_features, feature_sampler, scalar_fun, samp, feature_parameters)
end

#these call the above constructor
"""
$(TYPEDSIGNATURES)

Constructor for a `ScalarFeature` with cosine features
"""
function ScalarFourierFeature(
    n_features::Int,
    sampler::Sampler;
    feature_parameters::Dict{S} = Dict("sigma" => sqrt(2.0)),
) where {S <: AbstractString}
    return ScalarFeature(n_features, sampler, Cosine(); feature_parameters = feature_parameters)
end

"""
$(TYPEDSIGNATURES)

Constructor for a `ScalarFeature` with activation-function features (default ReLU)
"""
function ScalarNeuronFeature(
    n_features::Int,
    sampler::Sampler;
    activation_fun::SA = Relu(),
    kwargs...,
) where {SA <: ScalarActivation}
    return ScalarFeature(n_features, sampler, activation_fun; kwargs...)
end

"""
$(TYPEDSIGNATURES)

builds features (possibly batched) from an input matrix of size (input dimension, number of samples) output of dimension (number of samples, 1, number features) 
"""
function build_features(
    rf::ScalarFeature,
    inputs::M, # input_dim x n_sample
    batch_feature_idx::V,
) where {M <: AbstractMatrix, V <: AbstractVector}
    #    inputs = permutedims(inputs_t, (2, 1)) # n_sample x input_dim

    # build: sigma * scalar_function(xi . input + b)
    samp = get_feature_sample(rf)
    xi = get_distribution(samp)["xi"][:, batch_feature_idx] # dim_inputs x n_features
    #    features = inputs * xi[:, batch_feature_idx] # n_samples x n_features
    @tullio features[n, b] := inputs[d, n] * xi[d, b] # n_samples x output_dim x n_feature_batch
    is_biased = "bias" ∈ get_name(samp)
    if is_biased
        bias = get_distribution(samp)["bias"][1, batch_feature_idx] # 1 x n_features
        @tullio features[n, b] += bias[b]
    end

    sf = get_scalar_function(rf)
    features .= apply_scalar_function.(Ref(sf), features) # BOTTLENECK OF build_features.

    sigma = get_feature_parameters(rf)["sigma"] # scalar
    @. features *= sigma

    #consistent output shape with vector case, by putting output_dim = 1 in middle dimension
    return reshape(features, size(features, 1), 1, size(features, 2))
end

build_features(rf::ScalarFeature, inputs::M) where {M <: AbstractMatrix} =
    build_features(rf, inputs, collect(1:get_n_features(rf)))

"""
$(TYPEDSIGNATURES)

gets the output dimension (equals 1 for scalar-valued features)
"""
get_output_dim(rf::ScalarFeature) = 1
