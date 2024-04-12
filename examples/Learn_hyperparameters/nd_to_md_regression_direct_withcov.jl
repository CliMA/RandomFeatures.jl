# Example to learn hyperparameters of simple 1d-1d regression example.
# This example matches test/Methods/runtests.jl testset: "Fit and predict: 1D -> 1D"
# The (approximate) optimal values here are used in those tests.


using StableRNGs
using Distributions
using JLD2
using StatsBase
using LinearAlgebra
using Random
using Dates

PLOT_FLAG = true
println("plot flag:", PLOT_FLAG)
if PLOT_FLAG
    using Plots
end
using EnsembleKalmanProcesses
const EKP = EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Localizers

using RandomFeatures.ParameterDistributions
using RandomFeatures.DataContainers
using RandomFeatures.Samplers
using RandomFeatures.Features
using RandomFeatures.Methods
using RandomFeatures.Utilities

seed = 2024
ekp_seed = 99999
rng = StableRNG(seed)

function flat_to_chol(x::AbstractArray)
    choldim = Int(floor(sqrt(2 * length(x))))
    cholmat = zeros(choldim, choldim)
    for i in 1:choldim
        for j in 1:i
            cholmat[i, j] = x[sum(0:(i - 1)) + j]
        end
    end
    return cholmat
end

function posdef_correct(mat::AbstractMatrix, tol = 1e8 * eps())
    mat += permutedims(mat, (2, 1))
    mat *= 0.5 # symmetrize
    return mat + (abs(minimum(eigvals(mat))) + tol) * I #add to diag
end

## Functions of use
function RFM_from_hyperparameters(
    rng::RNG,
    l::RorM,
    lambda::L,
    n_features::Int,
    batch_sizes::Dict{S, Int},
    input_dim::Int,
    output_dim::Int,
) where {
    RNG <: AbstractRNG,
    RorM <: Union{Real, AbstractVecOrMat},
    S <: AbstractString,
    L <: Union{AbstractMatrix, UniformScaling, Real},
}

    # l = [input_dim params + output_dim params]
    μ_c = 0.0

    if input_dim > 1 && output_dim > 1
        cholU = flat_to_chol(l[1:Int(0.5 * input_dim * (input_dim + 1))])
        cholV = flat_to_chol(
            l[(Int(0.5 * input_dim * (input_dim + 1)) + 1):Int(
                0.5 * input_dim * (input_dim + 1) + 0.5 * output_dim * (output_dim + 1),
            )],
        )
        U = l[end - 1] * (cholU * permutedims(cholU, (2, 1)) + l[end - 1] * I)
        V = l[end] * (cholV * permutedims(cholV, (2, 1)) + l[end] * I)
    elseif input_dim == 1 && output_dim > 1
        U = ones(1, 1)
        cholV = flat_to_chol(l[2:(Int(0.5 * output_dim * (output_dim + 1)) + 1)])
        V = l[1] * (cholV * permutedims(cholV, (2, 1)) + l[1] * I)
    elseif input_dim > 1 && output_dim == 1
        cholU = flat_to_chol(l[1:Int(0.5 * input_dim * (input_dim + 1))])
        U = l[end] * (holU * permutedims(cholU, (2, 1)) + l[end] * I)
        V = ones(1, 1)
    end

    M = zeros(input_dim, output_dim) # n x p mean

    if !isposdef(U)
        println("U not posdef")
        U = posdef_correct(U)
    end
    if !isposdef(V)
        println("V not posdef")
        V = posdef_correct(V)
    end
    representation = "covariance" # "covariance"
    if representation == "precision"
        Uinv = inv(U)
        Vinv = inv(V)
        if !isposdef(Uinv)
            println("Uinv not posdef")

            U = posdef_correct(Uinv)
        end
        if !isposdef(Vinv)
            println("Vinv not posdef")

            V = posdef_correct(Vinv)
        end
    elseif representation == "covariance"
        nothing
    else
        throw(ArgumentError("representation must be \"covariance\" else \"precision\". Got $representation"))
    end

    pd = ParameterDistribution(
        Dict(
            "distribution" => Parameterized(MatrixNormal(M, U, V)),
            "constraint" => repeat([no_constraint()], input_dim * output_dim),
            "name" => "xi",
        ),
    )

    feature_sampler = FeatureSampler(pd, output_dim, rng = rng)
    vff = VectorFourierFeature(n_features, output_dim, feature_sampler)

    return RandomFeatureMethod(vff, batch_sizes = batch_sizes, regularization = lambda)
end


function calculate_mean_cov_and_coeffs(
    rng::RNG,
    l::RorM,
    lambda::L,
    n_features::Int,
    batch_sizes::Dict{S, Int},
    io_pairs::PairedDataContainer,
    mean_store::Matrix{FT},
    cov_store::Array{FT, 3},
    buffer::Array{FT, 3};
    decomp_type::S = "chol",
) where {
    RNG <: AbstractRNG,
    FT <: AbstractFloat,
    S <: AbstractString,
    RorM <: Union{Real, AbstractVecOrMat},
    L <: Union{AbstractMatrix, UniformScaling, Real},
}

    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train

    # split data into train/test randomly
    itrain = get_inputs(io_pairs)[:, 1:n_train]
    otrain = get_outputs(io_pairs)[:, 1:n_train]
    io_train_cost = PairedDataContainer(itrain, otrain)
    itest = get_inputs(io_pairs)[:, (n_train + 1):end]
    otest = get_outputs(io_pairs)[:, (n_train + 1):end]
    input_dim = size(itrain, 1)
    output_dim = size(otrain, 1)

    # build and fit the RF
    rfm = RFM_from_hyperparameters(rng, l, lambda, n_features, batch_sizes, input_dim, output_dim)
    if decomp_type == "chol"
        fitted_features = fit(rfm, io_train_cost, decomposition_type = "cholesky")
    else
        fitted_features = fit(rfm, io_train_cost, decomposition_type = "svd")
    end
    test_batch_size = get_batch_size(rfm, "test")
    batch_inputs = batch_generator(itest, test_batch_size, dims = 2) # input_dim x batch_size

    #we want to calc 1/var(y-mean)^2 + lambda/m * coeffs^2 in the end
    #    pred_mean, pred_cov = predict(rfm, fitted_features, DataContainer(itest))
    predict!(rfm, fitted_features, DataContainer(itest), mean_store, cov_store, buffer)
    # sizes (output_dim x n_test), (output_dim x output_dim x n_test) 


    scaled_coeffs = 1 / sqrt(n_features) * norm(get_coeffs(fitted_features))

    if decomp_type == "chol"
        chol_fac = get_decomposition(get_feature_factors(fitted_features)).L
        complexity = 2 * sum(log(chol_fac[i, i]) for i in 1:size(chol_fac, 1))
    else
        svd_singval = get_decomposition(get_feature_factors(fitted_features)).S
        complexity = sum(log, svd_singval) # note this is log(abs(det))
    end
    complexity = sqrt(complexity) # complexity must be positive

    println("sample_complexity", complexity)
    return scaled_coeffs, complexity

end


function estimate_mean_and_coeffnorm_covariance(
    rng::RNG,
    l::RorM,
    lambda::L,
    n_features::Int,
    batch_sizes::Dict{S, Int},
    io_pairs::PairedDataContainer,
    n_samples::Int,
    y;
    repeats::Int = 1,
) where {
    RNG <: AbstractRNG,
    S <: AbstractString,
    RorM <: Union{Real, AbstractVecOrMat},
    L <: Union{AbstractMatrix, UniformScaling, Real},
}
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train
    output_dim = size(get_outputs(io_pairs), 1)

    means = zeros(output_dim, n_samples, n_test)
    mean_of_covs = zeros(output_dim, output_dim, n_test)
    moc_tmp = similar(mean_of_covs)
    mtmp = zeros(output_dim, n_test)
    buffer = zeros(n_test, output_dim, n_features)
    complexity = zeros(1, n_samples)
    coeffl2norm = zeros(1, n_samples)

    for i in 1:n_samples
        for j in 1:repeats
            c, cplxty =
                calculate_mean_cov_and_coeffs(rng, l, lambda, n_features, batch_sizes, io_pairs, mtmp, moc_tmp, buffer)
            # m output_dim x n_test
            # v output_dim x output_dim x n_test
            # c n_features
            # cplxty 1

            # update vbles needed for cov
            means[:, i, :] .+= (y - mtmp) ./ repeats
            coeffl2norm[1, i] += sqrt(sum(abs2, c)) / repeats
            complexity[1, i] += cplxty / repeats

            # update vbles needed for mean
            @. mean_of_covs += moc_tmp / (repeats * n_samples)

        end
    end
    means = permutedims(means, (3, 2, 1))
    mean_of_covs = permutedims(mean_of_covs, (3, 1, 2))

    approx_σ2 = zeros(n_test * output_dim, n_test * output_dim)
    blockmeans = zeros(n_test * output_dim, n_samples)
    for i in 1:n_test
        id = ((i - 1) * output_dim + 1):(i * output_dim)
        approx_σ2[id, id] = mean_of_covs[i, :, :] # this ordering, so we can take a mean/cov in dims = 2.
        blockmeans[id, :] = permutedims(means[i, :, :], (2, 1))
    end

    sample_mat = vcat(blockmeans, coeffl2norm, complexity)
    Γ = cov(sample_mat, dims = 2)

    if !isposdef(approx_σ2)
        println("approx_σ2 not posdef")
        approx_σ2 = posdef_correct(approx_σ2)
    end

    return Γ, approx_σ2


    return Γ, approx_σ2

end

function calculate_ensemble_mean_and_coeffnorm(
    rng::RNG,
    lvecormat::VorM,
    lambda::L,
    n_features::Int,
    batch_sizes::Dict{S, Int},
    io_pairs::PairedDataContainer,
    y;
    repeats::Int = 1,
) where {
    RNG <: AbstractRNG,
    S <: AbstractString,
    VorM <: AbstractVecOrMat,
    L <: Union{AbstractMatrix, UniformScaling, Real},
}
    if isa(lvecormat, AbstractVector)
        lmat = reshape(lvecormat, 1, :)
    else
        lmat = lvecormat
    end
    N_ens = size(lmat, 2)
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train
    output_dim = size(get_outputs(io_pairs), 1)

    means = zeros(output_dim, N_ens, n_test)
    mean_of_covs = zeros(output_dim, output_dim, n_test)
    buffer = zeros(n_test, output_dim, n_features)
    complexity = zeros(1, N_ens)
    coeffl2norm = zeros(1, N_ens)
    moc_tmp = similar(mean_of_covs)
    mtmp = zeros(output_dim, n_test)

    for i in collect(1:N_ens)
        for j in collect(1:repeats)
            l = lmat[:, i]
            c, cplxty =
                calculate_mean_cov_and_coeffs(rng, l, lambda, n_features, batch_sizes, io_pairs, mtmp, moc_tmp, buffer)
            # m output_dim x n_test
            # v output_dim x output_dim x n_test
            # c n_features
            means[:, i, :] += (y - mtmp) ./ repeats
            @. mean_of_covs += moc_tmp / (repeats * N_ens)
            coeffl2norm[1, i] += sqrt(sum(c .^ 2)) / repeats
            complexity[1, i] += cplxty / repeats
        end
    end
    means = permutedims(means, (3, 2, 1))
    mean_of_covs = permutedims(mean_of_covs, (3, 1, 2))
    blockcovmat = zeros(n_test * output_dim, n_test * output_dim)
    blockmeans = zeros(n_test * output_dim, N_ens)
    for i in 1:n_test
        id = ((i - 1) * output_dim + 1):(i * output_dim)
        blockcovmat[id, id] = mean_of_covs[i, :, :]
        blockmeans[id, :] = permutedims(means[i, :, :], (2, 1))
    end

    if !isposdef(blockcovmat)
        println("blockcovmat not posdef")
        blockcovmat = posdef_correct(blockcovmat)
    end

    return vcat(blockmeans, coeffl2norm, complexity), blockcovmat
end

@time begin
    ## Begin Script, define problem setting
    println("Begin script")
    date_of_run = Date(2024, 4, 10)

    input_dim = 1
    output_dim = 3
    println("Number of input dimensions: ", input_dim)
    println("Number of output dimensions: ", output_dim)

    function ftest_1d_to_3d(x::M) where {M <: AbstractMatrix}
        out = zeros(3, size(x, 2))
        out[1, :] = mapslices(column -> sin(norm([i * c for (i, c) in enumerate(column)])^2), x, dims = 1)
        out[2, :] = mapslices(column -> exp(-0.1 * norm([i * c for (i, c) in enumerate(column)])^2), x, dims = 1)
        out[3, :] = mapslices(
            column ->
                norm([i * c for (i, c) in enumerate(column)]) * sin(1 / norm([i * c for (i, c) in enumerate(column)])^2) -
                1,
            x,
            dims = 1,
        )
        return out
    end

    #problem formulation
    n_data = 100
    x = rand(rng, MvNormal(zeros(input_dim), I), n_data)

    # diagonal noise
    #    cov_mat = Diagonal((5e-2)^2 * ones(output_dim))
    #cov_mat = (5e-2)^2*I(output_dim)
    # correlated noise
    cov_mat = convert(Matrix, Tridiagonal((5e-3) * ones(2), (2e-2) * ones(3), (5e-3) * ones(2)))

    noise_dist = MvNormal(zeros(output_dim), cov_mat)
    noise = rand(rng, noise_dist, n_data)

    # simple regularization
    #lambda = exp((1 / output_dim) * sum(log.(eigvals(cov_mat)))) * I(output_dim)
    # more complex
    lambda = cov_mat


    y = ftest_1d_to_3d(x) + noise
    io_pairs = PairedDataContainer(x, y)

    ## Define Hyperpriors for EKP

    μ_l = 5.0
    σ_l = 5.0
    # prior for non radial problem
    n_l = Int(0.5 * input_dim * (input_dim + 1)) + Int(0.5 * output_dim * (output_dim + 1))
    n_l += (input_dim > 1 && output_dim > 1) ? 2 : 0

    prior_lengthscale = constrained_gaussian("lengthscale", μ_l, σ_l, 0.0, Inf, repeats = n_l)
    priors = prior_lengthscale
    println("number of hyperparameters to train: ", n_l)

    # estimate the noise from running many RFM sample costs at the mean values
    batch_sizes = Dict("train" => 500, "test" => 500, "feature" => 500)
    n_train = Int(floor(0.8 * n_data))
    n_test = n_data - n_train
    n_features_opt = Int(floor(2 * n_train))
    #n_features = Int(floor(2 * n_data))
    # RF will perform poorly when n_features is close to n_train
    @assert(!(n_features_opt == n_train)) #


    repeats = 1

    CALC_TRUTH = true

    println("RHKS norm type: norm of coefficients")

    if CALC_TRUTH
        sample_multiplier = 1

        n_samples = (n_test * output_dim + 2) * sample_multiplier
        println("Estimating output covariance with ", n_samples, " samples")
        internal_Γ, approx_σ2 = estimate_mean_and_coeffnorm_covariance(
            rng,
            repeat([μ_l], n_l), # take mean values
            lambda,
            n_features_opt,
            batch_sizes,
            io_pairs,
            n_samples,
            y[:, (n_train + 1):end],
            repeats = repeats,
        )

        save("calculated_truth_cov.jld2", "internal_Γ", internal_Γ, "approx_σ2", approx_σ2)
    else
        println("Loading truth covariance from file...")
        internal_Γ = load("calculated_truth_cov.jld2")["internal_Γ"]
        approx_σ2 = load("calculated_truth_cov.jld2")["approx_σ2"]
    end

    Γ = internal_Γ
    for i in 1:(n_test - 1)
        Γ[((i - 1) * output_dim + 1):(i * output_dim), ((i - 1) * output_dim + 1):(i * output_dim)] += lambda[:, :]
    end
    Γ[(n_test * output_dim + 1):end, (n_test * output_dim + 1):end] += I
    println(
        "Estimated variance. Tr(cov) = ",
        tr(Γ[1:(n_test * output_dim), 1:(n_test * output_dim)]),
        " + ",
        Γ[end - 1, end - 1],
        " + ",
        Γ[end, end],
    )
    println("is EKP noise positive definite? ", isposdef(Γ))
    #println("noise in observations: ", Γ)
    # Create EKI
    N_ens = 10 * input_dim
    N_iter = 10
    update_cov_step = Inf

    initial_params = construct_initial_ensemble(priors, N_ens; rng_seed = ekp_seed)
    params_init = transform_unconstrained_to_constrained(priors, initial_params)#[1, :]
    println("Prior gives parameters between: [$(minimum(params_init)),$(maximum(params_init))]")

    #=
    min_complexity =
        isa(lambda, UniformScaling) ? n_features_opt * log(lambda.λ) :
        n_features_opt / output_dim * 2 * sum(log.(diag(cholesky(lambda).L)))
    min_complexity = sqrt(abs(min_complexity))


    data = vcat(reshape(y[:, (n_train + 1):end], :, 1), 0.0, min_complexity) #flatten data
    println("min_complexity: ", min_complexity)
    =#
    data = zeros(size(Γ, 1))


    ekiobj = [EKP.EnsembleKalmanProcess(initial_params, data[:], Γ, Inversion())]
    err = zeros(N_iter)
    println("Begin EKI iterations:")
    Δt = [1.0]

    for i in 1:N_iter

        #get parameters:
        lvec = transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1]))
        g_ens, _ = calculate_ensemble_mean_and_coeffnorm(
            rng,
            lvec,
            lambda,
            n_features_opt,
            batch_sizes,
            io_pairs,
            y[:, (n_train + 1):end],
            repeats = repeats,
        )

        if i % update_cov_step == 0 # to update cov if required

            constrained_u = transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1]))
            println("Estimating output covariance with ", n_samples, " samples")
            internal_Γ_new, approx_σ2_new = estimate_mean_and_coeffnorm_covariance(
                rng,
                mean(constrained_u, dims = 2)[:, 1], # take mean values
                lambda,
                n_features_opt,
                batch_sizes,
                io_pairs,
                n_samples,
                y[:, (n_train + 1):end],
                repeats = repeats,
            )
            Γ_new = internal_Γ_new
            # Γ_new[1:(n_test * output_dim), 1:(n_test * output_dim)] += approx_σ2_new
            # Γ_new[(n_test * output_dim + 1):end, (n_test * output_dim + 1):end] += I
            println(
                "Estimated variance. Tr(cov) = ",
                tr(Γ_new[1:n_test, 1:n_test]),
                " + ",
                tr(Γ_new[(n_test * output_dim + 1):end, (n_test * output_dim + 1):end]),
            )

            ekiobj[1] = EKP.EnsembleKalmanProcess(get_u_final(ekiobj[1]), data[:], Γ_new, Inversion())

        end


        EKP.update_ensemble!(ekiobj[1], g_ens, Δt_new = Δt[1])
        err[i] = get_error(ekiobj[1])[end] #mean((params_true - mean(params_i,dims=2)).^2)
        constrained_u = transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1]))
        println(
            "Iteration: " *
            string(i) *
            ", Error: " *
            string(err[i]) *
            ", for parameter means: \n" *
            string(mean(constrained_u, dims = 2)),
            "\n and sd :\n" * string(sqrt.(var(constrained_u, dims = 2))),
        )
        Δt[1] *= 1.0

    end



    #run actual experiment
    # override following parameters for actual run
    n_data_test = 100 * input_dim
    n_features_test = Int(floor(2 * n_data_test))
    println("number of training data: ", n_data_test)
    println("number of features: ", n_features_test)
    x_test = rand(rng, MvNormal(zeros(input_dim), 0.5 * I), n_data_test)
    noise_test = rand(rng, noise_dist, n_data_test)

    y_test = ftest_1d_to_3d(x_test) + noise_test
    io_pairs_test = PairedDataContainer(x_test, y_test)

    # get feature distribution
    final_lvec = get_ϕ_mean_final(priors, ekiobj[1])
    println("**********")
    println("Optimal lengthscales: $(final_lvec)")
    println("**********")


    rfm = RFM_from_hyperparameters(rng, final_lvec, lambda, n_features_test, batch_sizes, input_dim, output_dim)
    fitted_features = fit(rfm, io_pairs_test, decomposition_type = "cholesky")

    if PLOT_FLAG
        # learning on Normal(0,1) dist, forecast on (-2,2)
        xrange = reshape(collect(-2.01:0.02:2.01), 1, :)

        yrange = ftest_1d_to_3d(xrange)

        pred_mean_slice, pred_cov_slice = predict(rfm, fitted_features, DataContainer(xrange))

        for i in 1:output_dim
            pred_cov_slice[i, i, :] = max.(pred_cov_slice[i, i, :], 0.0)
        end

        figure_save_directory = joinpath(@__DIR__, "output", string(date_of_run))
        if !isdir(figure_save_directory)
            mkpath(figure_save_directory)
        end

        #plot diagonal

        xplot = xrange[:]
        plt = plot(
            xplot,
            yrange[1, :],
            show = false,
            color = "black",
            linewidth = 5,
            size = (600, 600),
            legend = :topleft,
            label = "Target",
        )
        plot!(
            xplot,
            yrange[2, :],
            show = false,
            color = "black",
            linewidth = 5,
            size = (600, 600),
            legend = :topleft,
            label = "Target",
        )
        plot!(
            xplot,
            yrange[3, :],
            show = false,
            color = "black",
            linewidth = 5,
            size = (600, 600),
            legend = :topleft,
            label = "Target",
        )
        scatter!(x_test[:], y_test[1, :], color = "blue", label = "", marker = :x)

        plot!(
            xplot,
            pred_mean_slice[1, :],
            ribbon = [2 * sqrt.(pred_cov_slice[1, 1, :]); 2 * sqrt.(pred_cov_slice[1, 1, :])],
            label = "Fourier",
            color = "blue",
        )
        scatter!(x_test[:], y_test[2, :], color = "red", label = "", marker = :x)
        plot!(
            xplot,
            pred_mean_slice[2, :],
            ribbon = [2 * sqrt.(pred_cov_slice[2, 2, :]); 2 * sqrt.(pred_cov_slice[2, 2, :])],
            label = "Fourier",
            color = "red",
        )
        scatter!(x_test[:], y_test[3, :], color = "green", label = "", marker = :x)
        plot!(
            xplot,
            pred_mean_slice[3, :],
            ribbon = [2 * sqrt.(pred_cov_slice[3, 3, :]); 2 * sqrt.(pred_cov_slice[3, 3, :])],
            label = "Fourier",
            color = "green",
        )

        savefig(plt, joinpath(figure_save_directory, "Fit_and_predict_1D_to_MD.pdf"))
        savefig(plt, joinpath(figure_save_directory, "Fit_and_predict_1D_to_MD.png"))



    end

end
