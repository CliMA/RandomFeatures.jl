export VectorFeature, VectorFourierFeature, VectorNeuronFeature
export build_features

"""
$(TYPEDEF)

Contains information to build and sample RandomFeatures mapping from N-D -> M-D

$(TYPEDFIELDS)
"""
struct VectorFeature{S <: AbstractString, SF <: ScalarFunction} <: RandomFeature
    "Number of features"
    n_features::Int
    "Dimension of output"
    output_dim::Int
    "Sampler of the feature distribution"
    feature_sampler::Sampler
    "ScalarFunction mapping R -> R"
    scalar_function::SF
    "Current `Sample` from sampler"
    feature_sample::ParameterDistribution
    "hyperparameters in Feature (and not in Sampler)"
    feature_parameters::Union{Dict{String}, Nothing}
end

# common constructors
"""
$(TYPEDSIGNATURES)

basic constructor for a `VectorFeature'
"""
function VectorFeature(
    n_features::Int,
    output_dim::Int,
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

    return VectorFeature{S, SF}(n_features, output_dim, feature_sampler, scalar_fun, samp, feature_parameters)
end

#these call the above constructor
"""
$(TYPEDSIGNATURES)

Constructor for a `VectorFeature` with cosine features
"""
function VectorFourierFeature(
    n_features::Int,
    output_dim::Int,
    sampler::Sampler;
    feature_parameters::Dict{S} = Dict("sigma" => sqrt(2.0)),
) where {S <: AbstractString}
    return VectorFeature(n_features, output_dim, sampler, Cosine(); feature_parameters = feature_parameters)
end

"""
$(TYPEDSIGNATURES)

Constructor for a `VectorFeature` with activation-function features (default ReLU)
"""
function VectorNeuronFeature(
    n_features::Int,
    output_dim::Int,
    sampler::Sampler;
    activation_fun::ScalarActivation = Relu(),
    kwargs...,
)
    return VectorFeature(n_features, output_dim, sampler, activation_fun; kwargs...)
end

"""
$(TYPEDSIGNATURES)

builds features (possibly batched) from an input matrix of size (input dimension,number of samples) output of dimension (number of samples, output dimension, number features) 
"""
function build_features(
    rf::VectorFeature,
    inputs::M, # input_dim x n_sample
    batch_feature_idx::V,
) where {M <: AbstractMatrix, V <: AbstractVector}
    # build: sigma * scalar_function(xi * input + b)
    samp = get_feature_sample(rf)

    #TODO: What we want:
    # xi = get_distribution(samp)["xi"][:,:,batch_feature_idx] # input_dim x output_dim x n_feature_batch
    # for now, as matrix distributions aren't yet supported, xi is flattened, so we reshape
    xi_flat = get_distribution(samp)["xi"][:, batch_feature_idx] # (input_dim x output_dim) x n_feature_batch
    sampler = get_feature_sampler(rf)
    pd = get_parameter_distribution(sampler)
    xi_size = size(get_distribution(pd)["xi"])
    if length(xi_size) > 1
        xi = reshape(xi_flat, (xi_size[1], xi_size[2], size(xi_flat, 2))) # in x out x n_feature_batch
    else
        xi = reshape(xi_flat, xi_size[1], 1, size(xi_flat, 2)) # in x out x n_feature_batch
    end

    features = zeros(size(inputs, 2), size(xi, 2), size(xi, 3))
    @tullio features[n, p, b] = inputs[d, n] * xi[d, p, b] # n_samples x output_dim x n_feature_batch

    is_biased = "bias" ∈ get_name(samp)
    if is_biased
        bias = get_distribution(samp)["bias"][:, batch_feature_idx] # dim_output x n_features
        @tullio features[n, p, b] += bias[p, b]
    end

    sf = get_scalar_function(rf)
    features .= apply_scalar_function.(Ref(sf), features)

    sigma = get_feature_parameters(rf)["sigma"] # scalar
    features .*= sigma

    return features # n_sample x n_feature_batch x output_dim
end

build_features(rf::VectorFeature, inputs::M) where {M <: AbstractMatrix} =
    build_features(rf, inputs, collect(1:get_n_features(rf)))

"""
$(TYPEDSIGNATURES)

gets the output dimension (equals 1 for scalar-valued features)
"""
get_output_dim(rf::VectorFeature) = rf.output_dim
