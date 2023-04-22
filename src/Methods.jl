module Methods

import StatsBase: sample, fit, predict, predict!

using LinearAlgebra,
    DocStringExtensions,
    RandomFeatures.Features,
    RandomFeatures.Utilities,
    EnsembleKalmanProcesses.DataContainers,
    Tullio,
    LoopVectorization

export RandomFeatureMethod,
    Fit,
    get_random_feature,
    get_batch_sizes,
    get_batch_size,
    get_regularization,
    get_tullio_threading,
    sample,
    get_feature_factors,
    get_coeffs,
    fit,
    predict,
    predict!,
    predictive_mean,
    predictive_cov,
    predictive_mean!,
    predictive_cov!,
    predict_prior,
    predict_prior_mean,
    predict_prior_cov

"""
$(TYPEDEF)

Holds configuration for the random feature fit

$(TYPEDFIELDS)
"""
struct RandomFeatureMethod{S <: AbstractString, USorM <: Union{UniformScaling, AbstractMatrix}}
    "The random feature object"
    random_feature::RandomFeature
    "A dictionary specifying the batch sizes. Must contain \"train\", \"test\", and \"feature\" keys"
    batch_sizes::Dict{S, Int}
    "A positive definite matrix used during the fit method to regularize the linear solve"
    regularization::USorM
    "Use multithreading provided by Tullio"
    tullio_threading::Bool
end

"""
$(TYPEDSIGNATURES)

Basic constructor for a `RandomFeatureMethod`. 
"""
function RandomFeatureMethod(
    random_feature::RandomFeature;
    regularization::USorMorR = 1e12 * eps() * I,
    batch_sizes::Dict{S, Int} = Dict("train" => 0, "test" => 0, "feature" => 0),
    tullio_threading = true,
) where {S <: AbstractString, USorMorR <: Union{<:Real, AbstractMatrix{<:Real}, UniformScaling}}

    if !all([key ∈ keys(batch_sizes) for key in ["train", "test", "feature"]])
        throw(ArgumentError("batch_sizes keys must contain all of \"train\", \"test\", and \"feature\""))
    end
    if isa(regularization, Real)
        if regularization < 0
            @info "input regularization < 0 is invalid, using regularization = 1e12*eps()"
            lambda = 1e12 * eps() * I
        else
            lambda = regularization * I
        end
    else
        if !isposdef(regularization) #check positive definiteness
            tol = 1e12 * eps() #MAGIC NUMBER
            lambda = posdef_correct(regularization, tol = tol)
            @warn "input regularization matrix is not positive definite, replacing with nearby positive definite matrix with minimum eigenvalue $tol"
        else

            lambda = regularization
        end
    end

    return RandomFeatureMethod{S, typeof(lambda)}(random_feature, batch_sizes, lambda, tullio_threading)
end

"""
$(TYPEDSIGNATURES)

gets the `random_feature` field
"""
get_random_feature(rfm::RandomFeatureMethod) = rfm.random_feature

"""
$(TYPEDSIGNATURES)

gets the `batch_sizes` field
"""
get_batch_sizes(rfm::RandomFeatureMethod) = rfm.batch_sizes

"""
$(TYPEDSIGNATURES)

gets the `regularization` field
"""
get_regularization(rfm::RandomFeatureMethod) = rfm.regularization

"""
$(TYPEDSIGNATURES)

gets the `tullio_threading` field
"""
get_tullio_threading(rfm::RandomFeatureMethod) = rfm.tullio_threading

"""
$(TYPEDSIGNATURES)

samples the random_feature field
"""
sample(rfm::RandomFeatureMethod) = sample(get_random_feature(rfm))


"""
$(TYPEDSIGNATURES)

get the specified batch size from `batch_sizes` field
"""
get_batch_size(rfm::RandomFeatureMethod, key::S) where {S <: AbstractString} = get_batch_sizes(rfm)[key]


"""
$(TYPEDEF)

Holds the coefficients and matrix decomposition that describe a set of fitted random features.

$(TYPEDFIELDS)
"""
struct Fit{V <: AbstractVector, USorM <: Union{UniformScaling, AbstractMatrix}}
    "The `LinearAlgreba` matrix decomposition of `(1 / m) * Feature^T * Feature + regularization`"
    feature_factors::Decomposition
    "Coefficients of the fit to data"
    coeffs::V
    "feature-space regularization used during fit"
    regularization::USorM
end

"""
$(TYPEDSIGNATURES)

gets the `feature_factors` field
"""
get_feature_factors(f::Fit) = f.feature_factors


"""
$(TYPEDSIGNATURES)

gets the `coeffs` field
"""
get_coeffs(f::Fit) = f.coeffs

"""
$(TYPEDSIGNATURES)

gets the `regularization` field (note this is the feature-space regularization)
"""
get_regularization(f::Fit) = f.regularization



"""
$(TYPEDSIGNATURES)

Fits a `RandomFeatureMethod` to input-output data, optionally provide a preferred `LinearAlgebra` matrix decomposition.
Returns a `Fit` object.
"""
function fit(
    rfm::RandomFeatureMethod,
    input_output_pairs::PairedDataContainer;
    decomposition_type::S = "cholesky",
) where {S <: AbstractString}

    (input, output) = get_data(input_output_pairs)
    input_dim, n_data = size(input)
    output_dim = size(output, 1) # for scalar features this is 1

    train_batch_size = get_batch_size(rfm, "train")
    rf = get_random_feature(rfm)
    tullio_threading = get_tullio_threading(rfm)
    n_features = get_n_features(rf)
    #data are columns, batch over samples

    lambda = get_regularization(rfm)
    build_regularization = !isa(lambda, UniformScaling) # build if lambda is not a uniform scaling
    if build_regularization
        lambda_new = zeros(n_features, n_features)
    else
        lambda_new = lambda
    end
    Phi = build_features(rf, input)
    FT = eltype(Phi)

    # regularization build needed with p.d matrix lambda
    if build_regularization
        if output_dim * n_data < n_features
            lambda_new = exp(1.0 / output_dim * log(det(lambda))) * I #det(X)^{1/m}
            @info(
                "pos-def regularization formulation ill-defined for output_dim ($output_dim) * n_data ($n_data) < n_feature ($n_features). \n Treating it as if regularization was a uniform scaling size det(regularization_matrix)^{1 / output_dim}: $(lambda_new.λ)"
            )
        else

            if !isa(lambda, Diagonal)
                if !tullio_threading
                    @tullio threads = false lambdaT_times_phi[n, p, i] := lambda[q, p] * Phi[n, q, i] # (I ⊗ Λᵀ) Φ
                else
                    @tullio lambdaT_times_phi[n, p, i] := lambda[q, p] * Phi[n, q, i] # (I ⊗ Λᵀ) Φ
                end
            else
                lam_diag = [lambda[i, i] for i in 1:size(lambda, 1)]

                if !tullio_threading
                    @tullio threads = false lambdaT_times_phi[n, p, i] := lam_diag[p] * Phi[n, p, i] # (I ⊗ Λᵀ) Φ
                else
                    @tullio lambdaT_times_phi[n, p, i] := lam_diag[p] * Phi[n, p, i] # (I ⊗ Λᵀ) Φ                
                end
            end
            #reshape to stacks columns, i.e n,p -> np does (n,...,n) p times
            rhs = reshape(permutedims(lambdaT_times_phi, (2, 1, 3)), (n_data * output_dim, n_features))
            lhs = reshape(permutedims(Phi, (2, 1, 3)), (n_data * output_dim, n_features))

            # Solve Φ Bᵀ = (I ⊗ Λᵀ) Φ and transpose for B (=lambda_new)
            lhs_svd = svd(lhs) #stable solve of rank deficient systems with SVD
            th_idx = 1:sum(lhs_svd.S .> 1e-2 * maximum(lhs_svd.S))
            lambda_new =
                lhs_svd.V[:, th_idx] *
                Diagonal(1 ./ lhs_svd.S[th_idx]) *
                permutedims(lhs_svd.U[:, th_idx], (2, 1)) *
                rhs
            #make positive definite
            lambda_new = posdef_correct(lambda_new)

        end
    end

    PhiTY = zeros(n_features) #
    PhiTPhi = zeros(n_features, n_features)
    if !tullio_threading
        @tullio threads = false PhiTY[j] = Phi[n, p, j] * output[p, n]
        @tullio threads = false PhiTPhi[i, j] = Phi[n, p, i] * Phi[n, p, j] # BOTTLENECK
    else
        @tullio PhiTY[j] = Phi[n, p, j] * output[p, n]
        @tullio PhiTPhi[i, j] = Phi[n, p, i] * Phi[n, p, j] # BOTTLENECK
    end
    # alternative using svd - turns out to be slower and more mem intensive

    @. PhiTPhi /= FT(n_features)

    # solve the linear system
    # (PhiTPhi + lambda_new ) * beta = PhiTY

    if lambda_new == 0 * I
        feature_factors = Decomposition(PhiTPhi, "pinv")
    else
        # in-place add lambda (as we don't use PhiTPhi again after this)
        if isa(lambda_new, UniformScaling)
            # much quicker than just adding lambda_new...
            for i in 1:size(PhiTPhi, 1)
                PhiTPhi[i, i] += lambda_new.λ
            end
        else
            @. PhiTPhi += lambda_new
        end
        feature_factors = Decomposition(PhiTPhi, decomposition_type)
        # bottleneck for small problems only (much quicker than PhiTPhi for big problems)

    end

    coeffs = linear_solve(feature_factors, PhiTY, tullio_threading = tullio_threading) #n_features x n_samples x dim_output

    return Fit{typeof(coeffs), typeof(lambda_new)}(feature_factors, coeffs, lambda_new)
end


"""
    $(TYPEDSIGNATURES)
        
    Makes a prediction of mean and (co)variance of fitted features on new input data
"""
function predict(rfm::RandomFeatureMethod, fit::Fit, new_inputs::DataContainer; kwargs...)
    pred_mean, features = predictive_mean(rfm, fit, new_inputs; kwargs...)
    pred_cov = predictive_cov(rfm, fit, new_inputs, features; kwargs...)

    return pred_mean, pred_cov
end

"""
$(TYPEDSIGNATURES)
        
Makes a prediction of mean and (co)variance of fitted features on new input data, overwriting the provided stores.
- mean_store:`output_dim` x `n_samples`
- cov_store:`output_dim` x `output_dim` x `n_samples`
- buffer:`n_samples` x `output_dim` x `n_features`
"""
function predict!(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    mean_store::M,
    cov_store::A,
    buffer::A;
    kwargs...,
) where {M <: AbstractMatrix{<:AbstractFloat}, A <: AbstractArray{<:AbstractFloat, 3}}
    #build features once only
    features = predictive_mean!(rfm, fit, new_inputs, mean_store; kwargs...)
    predictive_cov!(rfm, fit, new_inputs, cov_store, buffer, features; kwargs...)
    nothing
end


"""
$(TYPEDSIGNATURES)

Makes a prediction of mean and (co)variance with unfitted features on new input data
"""
function predict_prior(rfm::RandomFeatureMethod, new_inputs::DataContainer; kwargs...)
    prior_mean, features = predict_prior_mean(rfm, new_inputs; kwargs...)
    prior_cov = predict_prior_cov(rfm, new_inputs, features; kwargs...)
    return prior_mean, prior_cov
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of mean with unfitted features on new input data
"""
function predict_prior_mean(rfm::RandomFeatureMethod, new_inputs::DataContainer; kwargs...)
    rf = get_random_feature(rfm)
    n_features = get_n_features(rf)
    coeffs = ones(n_features)
    return predictive_mean(rfm, coeffs, new_inputs; kwargs...)
end

function predict_prior_mean(
    rfm::RandomFeatureMethod,
    new_inputs::DataContainer,
    prebuilt_features::A;
    kwargs...,
) where {A <: AbstractArray{<:AbstractFloat, 3}}
    rf = get_random_feature(rfm)
    n_features = get_n_features(rf)
    coeffs = ones(n_features)
    return predictive_mean(rfm, coeffs, new_inputs, prebuilt_features; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of (co)variance with unfitted features on new input data
"""
function predict_prior_cov(rfm::RandomFeatureMethod, new_inputs::DataContainer; kwargs...)
    inputs = get_data(new_inputs)
    rf = get_random_feature(rfm)
    features = build_features(rf, inputs) # bsize x output_dim x n_features
    return predict_prior_cov(rfm, new_inputs, features; kwargs...), features
end

function predict_prior_cov(
    rfm::RandomFeatureMethod,
    new_inputs::DataContainer,
    prebuilt_features::A;
    tullio_threading = true,
    kwargs...,
) where {A <: AbstractArray{<:AbstractFloat, 3}}
    #TODO optimize with woodbury as with other predictive_cov
    inputs = get_data(new_inputs)

    test_batch_size = get_batch_size(rfm, "test")
    rf = get_random_feature(rfm)
    output_dim = get_output_dim(rf)
    n_features = get_n_features(rf)
    FT = eltype(prebuilt_features)
    if !tullio_threading
        @tullio threads = false cov_outputs[p, q, n] :=
            prebuilt_features[n, p, m] * prebuilt_features[l, q, m] - prebuilt_features[l, q, m] # output_dim, output_dim, size(inputs, 2)
    else
        @tullio cov_outputs[p, q, n] :=
            prebuilt_features[n, p, m] * prebuilt_features[l, q, m] - prebuilt_features[l, q, m] # output_dim, output_dim, size(inputs, 2)
    end
    @. cov_outputs /= FT(n_features)
    return cov_outputs
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of mean of fitted features on new input data.
Returns a `output_dim` x `n_samples` array.
"""
predictive_mean(rfm::RandomFeatureMethod, fit::Fit, new_inputs::DataContainer; kwargs...) =
    predictive_mean(rfm, get_coeffs(fit), new_inputs; kwargs...)

predictive_mean(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    prebuilt_features::A;
    kwargs...,
) where {A <: AbstractArray{<:AbstractFloat, 3}} =
    predictive_mean(rfm, get_coeffs(fit), new_inputs, prebuilt_features; kwargs...)

function predictive_mean(
    rfm::RandomFeatureMethod,
    coeffs::V,
    new_inputs::DataContainer;
    kwargs...,
) where {V <: AbstractVector}
    inputs = get_data(new_inputs)
    rf = get_random_feature(rfm)
    features = build_features(rf, inputs)
    return predictive_mean(rfm, coeffs, new_inputs, features; kwargs...), features
end

function predictive_mean(
    rfm::RandomFeatureMethod,
    coeffs::V,
    new_inputs::DataContainer,
    prebuilt_features::A;
    kwargs...,
) where {V <: AbstractVector{<:AbstractFloat}, A <: AbstractArray{<:AbstractFloat, 3}}

    inputs = get_data(new_inputs)
    rf = get_random_feature(rfm)
    n_samples = size(inputs, 2)
    output_dim = get_output_dim(rf)
    mean_store = zeros(output_dim, n_samples)
    predictive_mean!(rfm, coeffs, new_inputs, mean_store, prebuilt_features; kwargs...)
    return mean_store
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of mean of fitted features on new input data.
Writes into a provided `output_dim` x `n_samples` array: `mean_store`.
"""
predictive_mean!(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    mean_store::M;
    kwargs...,
) where {M <: Matrix{<:AbstractFloat}} = predictive_mean!(rfm, get_coeffs(fit), new_inputs, mean_store; kwargs...)

predictive_mean!(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    mean_store::M,
    features::A;
    kwargs...,
) where {M <: Matrix{<:AbstractFloat}, A <: AbstractArray{<:AbstractFloat, 3}} =
    predictive_mean!(rfm, get_coeffs(fit), new_inputs, mean_store, features; kwargs...)


function predictive_mean!(
    rfm::RandomFeatureMethod,
    coeffs::V,
    new_inputs::DataContainer,
    mean_store::M;
    kwargs...,
) where {V <: AbstractVector{<:AbstractFloat}, M <: Matrix{<:AbstractFloat}}
    inputs = get_data(new_inputs)
    rf = get_random_feature(rfm)
    features = build_features(rf, inputs)
    predictive_mean!(rfm, coeffs, new_inputs, mean_store, features; kwargs...)
    return features
end

function predictive_mean!(
    rfm::RandomFeatureMethod,
    coeffs::V,
    new_inputs::DataContainer,
    mean_store::M,
    prebuilt_features::A;
    tullio_threading = true,
    kwargs...,
) where {V <: AbstractVector{<:AbstractFloat}, M <: Matrix{<:AbstractFloat}, A <: AbstractArray{<:AbstractFloat, 3}}
    inputs = get_data(new_inputs)
    rf = get_random_feature(rfm)
    tullio_threading = get_tullio_threading(rfm)
    output_dim = get_output_dim(rf)
    n_features = get_n_features(rf)
    if !(size(mean_store) == (output_dim, size(inputs, 2)))
        throw(
            DimensionMismatch(
                "provided storage for output expected to be size ($(output_dim),$(size(inputs,2))) got $(size(mean_store))",
            ),
        )
    end
    if !tullio_threading
        @tullio threads = false mean_store[p, n] = prebuilt_features[n, p, m] * coeffs[m]
    else
        @tullio mean_store[p, n] = prebuilt_features[n, p, m] * coeffs[m]
    end
    FT = eltype(prebuilt_features)
    @. mean_store /= FT(n_features)

    nothing
end

"""
    $(TYPEDSIGNATURES)
    
    Makes a prediction of (co)variance of fitted features on new input data.
Returns a `output_dim` x `output_dim` x `n_samples` array
"""
function predictive_cov(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    prebuilt_features::A;
    kwargs...,
) where {A <: AbstractArray{<:AbstractFloat, 3}}

    inputs = get_data(new_inputs)
    n_samples = size(inputs, 2)
    rf = get_random_feature(rfm)
    output_dim = get_output_dim(rf)
    n_features = get_n_features(rf)
    cov_store = zeros(output_dim, output_dim, n_samples)
    buffer = zeros(n_samples, output_dim, n_features)
    predictive_cov!(rfm, fit, new_inputs, cov_store, buffer, prebuilt_features; kwargs...)
    return cov_store
end


function predictive_cov(rfm::RandomFeatureMethod, fit::Fit, new_inputs::DataContainer; kwargs...)

    rf = get_random_feature(rfm)
    inputs = get_data(new_inputs)
    features = build_features(rf, inputs) # build_features gives bsize x output_dim x n_features

    return predictive_cov(rfm, fit, new_inputs, features; kwargs...), features
end

"""
    $(TYPEDSIGNATURES)
    
    Makes a prediction of (co)variance of fitted features on new input data.
Writes into a provided `output_dim` x `output_dim` x `n_samples` array: `cov_store`, and uses provided `n_samples` x `output_dim` x `n_features` buffer.
"""
function predictive_cov!(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    cov_store::A,
    buffer::A,
    prebuilt_features::A;
    tullio_threading = true,
    kwargs...,
) where {A <: AbstractArray{<:AbstractFloat, 3}}

    # unlike in mean case, we must perform a linear solve for coefficients at every test point.
    # thus we return both the covariance and the input-dep coefficients
    # note the covariance here is a posterior variance in 1d outputs, it is not the posterior covariance
    inputs = get_data(new_inputs)

    test_batch_size = get_batch_size(rfm, "test")
    features_batch_size = get_batch_size(rfm, "feature")
    rf = get_random_feature(rfm)
    tullio_threading = get_tullio_threading(rfm)
    n_features = get_n_features(rf)
    lambda = get_regularization(fit)

    output_dim = get_output_dim(rf)
    coeffs = get_coeffs(fit)
    PhiTPhi_reg_factors = get_feature_factors(fit)
    inv_decomp = get_inv_decomposition(PhiTPhi_reg_factors)
    if !(size(cov_store) == (output_dim, output_dim, size(inputs, 2)))
        throw(
            DimensionMismatch(
                "provided storage for output expected to be size ($(output_dim),$(output_dim),$(size(inputs,2))) got $(size(cov_store))",
            ),
        )
    end

    FT = eltype(prebuilt_features)
    if isa(lambda, UniformScaling)
        if !(size(buffer) == (size(inputs, 2), output_dim, n_features))
            throw(
                DimensionMismatch(
                    "provided storage for tmp buffer expected to be size ($(size(inputs,2)),$(output_dim),$(n_features)), got $(size(buffer))",
                ),
            )
        end
        if !tullio_threading
            @tullio threads = false buffer[n, p, o] = prebuilt_features[n, p, m] * inv_decomp[m, o]
            @tullio threads = false cov_store[p, q, n] = buffer[n, p, o] * prebuilt_features[n, q, o]
        else
            @tullio buffer[n, p, o] = prebuilt_features[n, p, m] * inv_decomp[m, o]
            @tullio cov_store[p, q, n] = buffer[n, p, o] * prebuilt_features[n, q, o]
        end
        @. cov_store /= (FT(n_features) / lambda.λ) #i.e. * lambda / n_features

    else
        PhiTPhi_reg = get_full_matrix(PhiTPhi_reg_factors)
        @. PhiTPhi_reg -= lambda
        if !tullio_threading
            @tullio threads = false buffer[n, p, i] = PhiTPhi_reg[i, j] * prebuilt_features[n, p, j] # BOTTLENECK OF PREDICTION
            coeff_outputs = linear_solve(PhiTPhi_reg_factors, buffer, tullio_threading = tullio_threading) # n_features x bsize x output_dim
            # make sure we don't use := in following line, or it wont modify the input argument.
            @tullio threads = false cov_store[p, q, n] =
                prebuilt_features[n, p, m] * (prebuilt_features[n, q, m] - coeff_outputs[n, q, m])
        else
            @tullio buffer[n, p, i] = PhiTPhi_reg[i, j] * prebuilt_features[n, p, j] # BOTTLENECK OF PREDICTION
            coeff_outputs = linear_solve(PhiTPhi_reg_factors, buffer) # n_features x bsize x output_dim
            # make sure we don't use := in following line, or it wont modify the input argument.
            @tullio cov_store[p, q, n] =
                prebuilt_features[n, p, m] * (prebuilt_features[n, q, m] - coeff_outputs[n, q, m])
        end
        @. cov_store /= FT(n_features)

        # IMPORTANT, we remove lambda in-place for calculation only, so must add it back
        @. PhiTPhi_reg += lambda

    end
    nothing
end

function predictive_cov!(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    cov_store::A,
    buffer::A;
    kwargs...,
) where {A <: AbstractArray{<:AbstractFloat, 3}}

    rf = get_random_feature(rfm)
    inputs = get_data(new_inputs)
    features = build_features(rf, inputs)
    predictive_cov!(rfm, fit, new_inputs, cov_store, buffer, features; kwargs...)
    return features
end


# TODO
# function posterior_cov(rfm::RandomFeatureMethod, u_input, v_input)
# 
# end


end # module
