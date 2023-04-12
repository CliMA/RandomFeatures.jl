module Methods

import StatsBase: sample, fit, predict, predict!

using LinearAlgebra,
    DocStringExtensions,
    RandomFeatures.Features,
    RandomFeatures.Utilities,
    EnsembleKalmanProcesses.DataContainers,
    Tullio

export RandomFeatureMethod,
    Fit,
    get_random_feature,
    get_batch_sizes,
    get_batch_size,
    get_regularization,
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
end

"""
$(TYPEDSIGNATURES)

Basic constructor for a `RandomFeatureMethod`. 
"""
function RandomFeatureMethod(
    random_feature::RandomFeature;
    regularization::USorMorR = 1e12 * eps() * I,
    batch_sizes::Dict{S, Int} = Dict("train" => 0, "test" => 0, "feature" => 0),
) where {S <: AbstractString, USorMorR <: Union{Real, AbstractMatrix, UniformScaling}}

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

    return RandomFeatureMethod{S, typeof(lambda)}(random_feature, batch_sizes, lambda)
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
    decomposition_type::S = "svd",
) where {S <: AbstractString}

    (input, output) = get_data(input_output_pairs)
    input_dim, n_data = size(input)
    output_dim = size(output, 1) # for scalar features this is 1

    train_batch_size = get_batch_size(rfm, "train")
    rf = get_random_feature(rfm)
    n_features = get_n_features(rf)
    #data are columns, batch over samples

    batch_input = batch_generator(input, train_batch_size, dims = 2) # input_dim x batch_size
    batch_output = batch_generator(output, train_batch_size, dims = 2) # output_dim x batch_size

    lambda = get_regularization(rfm)
    build_regularization = !isa(lambda, UniformScaling) # build if lambda is not a uniform scaling
    if build_regularization
        lambda_new = zeros(n_features, n_features)
    else
        lambda_new = lambda
    end

    # regularization build needed with p.d matrix lambda
    if build_regularization
        if output_dim * n_data < n_features
            @info(
                "pos-def regularization formulation ill-defined for output_dim ($output_dim) * n_data ($n_data) < n_feature ($n_features). \n Treating it as if regularization was a uniform scaling size det(regularization_matrix)^{1 / output_dim}"
            )
            lambda_new = det(lambda)^(1 / output_dim) * I
        else

            # solve the rank-deficient linear system with SVD
            nonbatch_feature = build_features(rf, input) # n_data (N) x output_dim(P) x n_features(M)  

            lambdaT_times_phi = zeros(size(nonbatch_feature))
            @tullio lambdaT_times_phi[n, p, i] = lambda[q, p] * nonbatch_feature[n, q, i] # (I ⊗ Λᵀ) Φ

            #reshape to stacks columns, i.e n,p -> np does (n,...,n) p times
            rhs = reshape(permutedims(lambdaT_times_phi, (2, 1, 3)), (n_data * output_dim, n_features))
            lhs = reshape(permutedims(nonbatch_feature, (2, 1, 3)), (n_data * output_dim, n_features))

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

    for (ib, ob) in zip(batch_input, batch_output)
        batch_feature = build_features(rf, ib) # batch_size x output_dim x n_features  
        @tullio PhiTY[j] += batch_feature[n, p, j] * ob[p, n]
        @tullio PhiTPhi[i, j] += batch_feature[n, p, i] * batch_feature[n, p, j] # BOTTLENECK OF SOLVER
    end

    # alternative using svd - turns out to be slower and more mem intensive
    #=
    Phi = build_features(rf, input) # n_data x output_dim x n_features
    @tullio PhiTY[j] = Phi[n, p, j] * output[p, n]
    tmpPhi = zeros(size(Phi,1) * size(Phi,2),size(Phi,3))
    P = svd(@cast tmpPhi[(j,i),k] = Phi[i,j,k])
    out = similar(P.V)
    PhiTPhi = similar(out)
    mul!(PhiTPhi, mul!(out, P.V, Diagonal(P.S .^2)), P.Vt)
    =#

    @. PhiTPhi /= n_features
    PhiTY = reshape(PhiTY, n_features, 1, 1) # RHS kept as a 3-array (n_features x n_samples x dim_output)

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
    end
    coeffs = linear_solve(feature_factors, PhiTY) #n_features x n_samples x dim_output

    return Fit{typeof(coeffs[:]), typeof(lambda_new)}(feature_factors, coeffs[:], lambda_new)
end


"""
$(TYPEDSIGNATURES)

Makes a prediction of mean and (co)variance of fitted features on new input data
"""
function predict(rfm::RandomFeatureMethod, fit::Fit, new_inputs::DataContainer)
    pred_mean = predictive_mean(rfm, fit, new_inputs)
    pred_cov, _ = predictive_cov(rfm, fit, new_inputs)
    return pred_mean, pred_cov
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of mean and (co)variance of fitted features on new input data, overwriting the provided stores
"""
function predict!(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    mean_store::Matrix{R},
    cov_store::Array{R, 3},
) where {R <: Real}

    predictive_mean!(rfm, fit, new_inputs, mean_store)
    predictive_cov!(rfm, fit, new_inputs, cov_store)
    nothing
end


"""
$(TYPEDSIGNATURES)

Makes a prediction of mean and (co)variance with unfitted features on new input data
"""
function predict_prior(rfm::RandomFeatureMethod, new_inputs::DataContainer)
    return predict_prior_mean(rfm, new_inputs), predict_prior_cov(rfm, new_inputs)
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of mean with unfitted features on new input data
"""
function predict_prior_mean(rfm::RandomFeatureMethod, new_inputs::DataContainer)
    rf = get_random_feature(rfm)
    n_features = get_n_features(rf)
    coeffs = ones(n_features)
    return predictive_mean(rfm, coeffs, new_inputs)
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of (co)variance with unfitted features on new input data
"""
function predict_prior_cov(rfm::RandomFeatureMethod, new_inputs::DataContainer)
    inputs = get_data(new_inputs)

    test_batch_size = get_batch_size(rfm, "test")
    features_batch_size = get_batch_size(rfm, "feature")
    rf = get_random_feature(rfm)
    output_dim = get_output_dim(rf)
    n_features = get_n_features(rf)

    cov_outputs = zeros(output_dim, output_dim, size(inputs, 2)) # 

    batch_inputs = batch_generator(inputs, test_batch_size, dims = 2) # input_dim x batch_size
    batch_outputs = batch_generator(cov_outputs, test_batch_size, dims = 3) # 1 x batch_size

    for (ib, ob) in zip(batch_inputs, batch_outputs)
        features = build_features(rf, ib) # bsize x output_dim x n_features
        # here we do a pointwise calculation of var (1d output) for each test point
        #ob .+=
        #    sum(permutedims(features, (2, 1)) .* (permutedims(features, (2, 1)) - ones(size(features))'), dims = 1) /
        #    n_features
        # bsize x n_features
        @tullio ob[p, q, n] += features[n, p, m] * features[l, q, m] - features[l, q, m]
    end

    @. cov_outputs /= n_features
    return cov_outputs
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of mean of fitted features on new input data
"""
predictive_mean(rfm::RandomFeatureMethod, fit::Fit, new_inputs::DataContainer) =
    predictive_mean(rfm, get_coeffs(fit), new_inputs)

predictive_mean!(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    mean_store::Matrix{R},
) where {R <: Real} = predictive_mean!(rfm, get_coeffs(fit), new_inputs, mean_store)

function predictive_mean(rfm::RandomFeatureMethod, coeffs::V, new_inputs::DataContainer) where {V <: AbstractVector}

    inputs = get_data(new_inputs)

    test_batch_size = get_batch_size(rfm, "test")
    features_batch_size = get_batch_size(rfm, "feature")
    rf = get_random_feature(rfm)
    output_dim = get_output_dim(rf)
    outputs = zeros(output_dim, size(inputs, 2))

    n_features = get_n_features(rf)

    batch_inputs = batch_generator(inputs, test_batch_size, dims = 2) # input_dim x batch_size
    batch_outputs = batch_generator(outputs, test_batch_size, dims = 2) # input_dim x batch_size
    batch_coeffs = batch_generator(coeffs, features_batch_size) # batch_size
    batch_feature_idx = batch_generator(collect(1:n_features), features_batch_size) # batch_size

    for (ib, ob) in zip(batch_inputs, batch_outputs)
        for (cb, fb_i) in zip(batch_coeffs, batch_feature_idx)
            # features = build_features(rf, ib, fb_i) # n_samples x output_dim x n_features
            #Dot very important...
            #ob .+= permutedims(features * reshape(cb, :, 1) / n_features, (2, 1)) # 1 x n_samples
            @tullio ob[p, n] += build_features(rf, ib, fb_i)[n, p, m] * cb[m]
        end
    end
    @. outputs /= n_features

    return outputs
end

function predictive_mean!(
    rfm::RandomFeatureMethod,
    coeffs::V,
    new_inputs::DataContainer,
    mean_store::Matrix{R},
) where {R <: Real, V <: AbstractVector}

    inputs = get_data(new_inputs)

    test_batch_size = get_batch_size(rfm, "test")
    features_batch_size = get_batch_size(rfm, "feature")
    rf = get_random_feature(rfm)
    output_dim = get_output_dim(rf)
    if !(size(mean_store) == (output_dim, size(inputs, 2)))
        throw(
            DimensionMismatch(
                "provided storage for output expected to be size ($(output_dim),$(size(inputs,2))) got $(size(mean_store))",
            ),
        )
    end


    n_features = get_n_features(rf)

    batch_inputs = batch_generator(inputs, test_batch_size, dims = 2) # input_dim x batch_size
    batch_outputs = batch_generator(mean_store, test_batch_size, dims = 2) # input_dim x batch_size
    batch_coeffs = batch_generator(coeffs, features_batch_size) # batch_size
    batch_feature_idx = batch_generator(collect(1:n_features), features_batch_size) # batch_size
    for (ib, ob) in zip(batch_inputs, batch_outputs)
        for (idx_in, cb, fb_i) in zip(1:length(batch_coeffs), batch_coeffs, batch_feature_idx)
            #features = build_features(rf, ib, fb_i) # n_samples x output_dim x n_features
            #Dot very important...
            #ob .+= permutedims(features * reshape(cb, :, 1) / n_features, (2, 1)) # 1 x n_samples
            if idx_in == 1
                @tullio ob[p, n] = build_features(rf, ib, fb_i)[n, p, m] * cb[m]
            else
                @tullio ob[p, n] += build_features(rf, ib, fb_i)[n, p, m] * cb[m]
            end
        end
    end
    @. mean_store /= n_features
    nothing
end

"""
$(TYPEDSIGNATURES)

Makes a prediction of (co)variance of fitted features on new input data
"""
function predictive_cov(rfm::RandomFeatureMethod, fit::Fit, new_inputs::DataContainer)
    # unlike in mean case, we must perform a linear solve for coefficients at every test point.
    # thus we return both the covariance and the input-dep coefficients
    # note the covariance here is a posterior variance in 1d outputs, it is not the posterior covariance

    inputs = get_data(new_inputs)

    test_batch_size = get_batch_size(rfm, "test")
    features_batch_size = get_batch_size(rfm, "feature")
    rf = get_random_feature(rfm)
    n_features = get_n_features(rf)
    lambda = get_regularization(fit)

    output_dim = get_output_dim(rf)
    coeffs = get_coeffs(fit)
    PhiTPhi_reg_factors = get_feature_factors(fit)
    PhiTPhi_reg = get_full_matrix(PhiTPhi_reg_factors)

    cov_outputs = zeros(output_dim, output_dim, size(inputs, 2))
    coeff_outputs = zeros(n_features, size(inputs, 2), output_dim)

    batch_inputs = batch_generator(inputs, test_batch_size, dims = 2) # input_dim x batch_size
    batch_outputs = batch_generator(cov_outputs, test_batch_size, dims = 3) # 1 x batch_size
    batch_coeff_outputs = batch_generator(coeff_outputs, test_batch_size, dims = 2)

    if isa(lambda, UniformScaling)
        for (ib, ob, cob) in zip(batch_inputs, batch_outputs, batch_coeff_outputs)
            features = build_features(rf, ib) # bsize x output_dim x n_features  
            # "rhs[i, n, p] := features[n, p, i]"
            c_tmp = linear_solve(PhiTPhi_reg_factors, permutedims(features, (3, 1, 2))) # n_features x bsize x output_dim
            #Dot very important
            @tullio cob[i, n, p] += c_tmp[i, n, p]
            # here we do a pointwise calculation of var (1d output) for each test point
            #        ob .+= sum(permutedims(features, (2, 1)) .* (permutedims(features, (2, 1)) - c_tmp), dims = 1) / n_features
            @tullio ob[p, q, n] += features[n, p, m] * c_tmp[m, n, q]
        end
        @. cov_outputs /= (n_features / lambda.λ)

    else
        for (ib, ob, cob) in zip(batch_inputs, batch_outputs, batch_coeff_outputs)
            features = build_features(rf, ib) # bsize x output_dim x n_features  
            #rhs = PhiTPhi * permutedims(features, (2, 1)) # = 1/m * phi(X)^T * phi(X) * phi(x')^T = phi(X)^T * k(X,x')
            @tullio rhs[i, n, p] := (PhiTPhi_reg[i, j] - lambda[i, j]) * features[n, p, j] # BOTTLENECK OF PREDICTION
            c_tmp = linear_solve(PhiTPhi_reg_factors, rhs) # n_features x bsize x output_dim
            #Dot very important
            @tullio cob[i, n, p] += c_tmp[i, n, p]
            # here we do a pointwise calculation of var (1d output) for each test point
            #        ob .+= sum(permutedims(features, (2, 1)) .* (permutedims(features, (2, 1)) - c_tmp), dims = 1) / n_features
            @tullio ob[p, q, n] += features[n, p, m] * (features[n, q, m] - c_tmp[m, n, q])

        end
        @. cov_outputs /= n_features
    end

    return cov_outputs, coeff_outputs
end

function predictive_cov!(
    rfm::RandomFeatureMethod,
    fit::Fit,
    new_inputs::DataContainer,
    cov_store::Array{R, 3},
) where {R <: Real}

    # unlike in mean case, we must perform a linear solve for coefficients at every test point.
    # thus we return both the covariance and the input-dep coefficients
    # note the covariance here is a posterior variance in 1d outputs, it is not the posterior covariance

    inputs = get_data(new_inputs)

    test_batch_size = get_batch_size(rfm, "test")
    features_batch_size = get_batch_size(rfm, "feature")
    rf = get_random_feature(rfm)
    n_features = get_n_features(rf)
    lambda = get_regularization(fit)

    output_dim = get_output_dim(rf)
    coeffs = get_coeffs(fit)
    PhiTPhi_reg_factors = get_feature_factors(fit)
    PhiTPhi_reg = get_full_matrix(PhiTPhi_reg_factors)

    if !(size(cov_store) == (output_dim, output_dim, size(inputs, 2)))
        throw(
            DimensionMismatch(
                "provided storage for output expected to be size ($(output_dim),$(output_dim),$(size(inputs,2))) got $(size(cov_store))",
            ),
        )
    end

    #    cov_outputs = zeros(output_dim, output_dim, size(inputs, 2))

    batch_inputs = batch_generator(inputs, test_batch_size, dims = 2) # input_dim x batch_size
    batch_outputs = batch_generator(cov_store, test_batch_size, dims = 3) # 1 x batch_size

    if isa(lambda, UniformScaling)
        for (ib, ob) in zip(batch_inputs, batch_outputs)
            features = build_features(rf, ib) # bsize x output_dim x n_features  
            @tullio rhs[i, n, p] := features[n, p, i]
            c_tmp = linear_solve(PhiTPhi_reg_factors, rhs)#permutedims(features, (3,1,2))) # n_features x bsize x output_dim
            # here we do a pointwise calculation of var (1d output) for each test point
            #        ob .+= sum(permutedims(features, (2, 1)) .* (permutedims(features, (2, 1)) - c_tmp), dims = 1) / n_features
            @tullio ob[p, q, n] = features[n, p, m] * c_tmp[m, n, q]
        end
        @. cov_store /= (n_features / lambda.λ) #i.e. * lambda / n_features
    else

        for (ib, ob) in zip(batch_inputs, batch_outputs)
            features = build_features(rf, ib) # bsize x output_dim x n_features  
            #rhs = PhiTPhi * permutedims(features, (2, 1)) # = 1/m * phi(X)^T * phi(X) * phi(x')^T = phi(X)^T * k(X,x')
            @tullio rhs[i, n, p] := (PhiTPhi_reg[i, j] - lambda[i, j]) * features[n, p, j] # BOTTLENECK OF PREDICTION
            c_tmp = linear_solve(PhiTPhi_reg_factors, rhs) # n_features x bsize x output_dim
            # here we do a pointwise calculation of var (1d output) for each test point
            #        ob .+= sum(permutedims(features, (2, 1)) .* (permutedims(features, (2, 1)) - c_tmp), dims = 1) / n_features
            @tullio ob[p, q, n] = features[n, p, m] * (features[n, q, m] - c_tmp[m, n, q])

        end
        @. cov_store /= n_features
    end
    nothing
end


# TODO
# function posterior_cov(rfm::RandomFeatureMethod, u_input, v_input)
# 
# end


end # module
