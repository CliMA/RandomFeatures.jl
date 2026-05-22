export VectorFeature, VectorFourierFeature, VectorNeuronFeature
export build_features

"""
$(TYPEDEF)

Random feature model that maps N-dimensional inputs to M-dimensional vector outputs.

$(TYPEDFIELDS)

# Constructors

- `VectorFourierFeature(n_features, output_dim, sampler; feature_parameters)` — cosine (Fourier) features.
- `VectorNeuronFeature(n_features, output_dim, sampler; activation_fun = Relu())` — activation-function features.
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


"""
$(TYPEDSIGNATURES)

Return the output dimension of `rf`.
"""
get_output_dim(rf::VectorFeature) = rf.output_dim


# common constructors
"""
$(TYPEDSIGNATURES)

Construct a `VectorFeature` with `n_features` random features, `output_dim`-dimensional outputs,
sampled from `feature_sampler`, applying `scalar_fun` to compute feature values.

The `feature_sampler` parameter distribution must contain a named component `"xi"`.
`feature_parameters` must include the key `"sigma"` (output scale; default `1`).
"""
function VectorFeature(
    n_features::Int,
    output_dim::Int,
    feature_sampler::Sampler,
    scalar_fun::SF;
    feature_parameters::Dict{S} = Dict("sigma" => 1),
) where {S <: AbstractString, SF <: ScalarFunction}

    "xi" ∈ get_name(get_parameter_distribution(feature_sampler)) ||
        _throw_missing_xi(get_parameter_distribution(feature_sampler); where = :VectorFeature)

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

Construct a `VectorFeature` with cosine (Fourier) features mapping to `output_dim`-dimensional outputs.

The `sigma` feature parameter (default `sqrt(2)`) scales the output.
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

Construct a `VectorFeature` with a neural-network activation function (default `Relu`) mapping to `output_dim`-dimensional outputs.

Pass any `ScalarActivation` subtype as `activation_fun` to use a different activation.
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

Build random features from an `input_dim × n_samples` input matrix, evaluating only the feature
indices in `batch_feature_idx`.

Returns an `n_samples × output_dim × length(batch_feature_idx)` array.
"""
function build_features(
    rf::VectorFeature,
    inputs::M, # input_dim x n_sample
    batch_feature_idx::V,
) where {M <: AbstractMatrix, V <: AbstractVector}

    # build: sigma * scalar_function(xi * input + b)
    samp = get_feature_sample(rf)
    input_dim = size(inputs, 1)
    output_dim = get_output_dim(rf)
    #TODO: What we want:
    # xi = get_distribution(samp)["xi"][:,:,batch_feature_idx] # input_dim x output_dim x n_feature_batch
    # for now, as matrix distributions aren't yet supported, xi is flattened, so we reshape
    xi_flat = get_distribution(samp)["xi"][:, batch_feature_idx] # (input_dim x output_dim) x n_feature_batch
    sampler = get_feature_sampler(rf)
    pd = get_parameter_distribution(sampler)

    xi = reshape(xi_flat, input_dim, output_dim, size(xi_flat, 2))
    features = zeros(size(inputs, 2), size(xi, 2), size(xi, 3))
    @tullio features[n, p, b] = inputs[d, n] * xi[d, p, b] # n_samples x output_dim x n_feature_batch

    is_biased = "bias" ∈ get_name(samp)
    if is_biased
        bias = get_distribution(samp)["bias"][:, batch_feature_idx] # dim_output x n_features
        @tullio features[n, p, b] += bias[p, b]
    end

    sf = get_scalar_function(rf)
    features .= apply_scalar_function.(Ref(sf), features) # BOTTLENECK OF build_features.
    sigma = get_feature_parameters(rf)["sigma"] # scalar
    @. features *= sigma

    return features # n_sample x n_feature_batch x output_dim
end

build_features(rf::VectorFeature, inputs::M) where {M <: AbstractMatrix} =
    build_features(rf, inputs, collect(1:get_n_features(rf)))
