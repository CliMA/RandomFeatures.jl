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

using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.Localizers

using RandomFeatures.Samplers
using RandomFeatures.Features
using RandomFeatures.Methods
using RandomFeatures.Utilities

seed = 2024
ekp_seed = 99999
rng = StableRNG(seed)

## Functions of use
function RFM_from_hyperparameters(
    rng::AbstractRNG,
    l::Union{Real, AbstractVecOrMat},
    s::Real,
    regularizer::Real,
    n_features::Int,
    batch_sizes::Dict,
    input_dim::Int,
)

    μ_c = 0.0
    if isa(l, Real)
        σ_c = fill(l, input_dim)
    elseif isa(l, AbstractVector)
        if length(l) == 1
            σ_c = fill(l[1], input_dim)
        else
            σ_c = l
        end
    else
        isa(l, AbstractMatrix)
        σ_c = l[:, 1]
    end

    pd = ParameterDistribution(
        Dict(
            "distribution" => VectorOfParameterized(map(sd -> Normal(μ_c, sd), σ_c)),
            "constraint" => repeat([no_constraint()], input_dim),
            "name" => "xi",
        ),
    )
    feature_sampler = FeatureSampler(pd, rng = rng)
    # Learn hyperparameters for different feature types

    sf = ScalarFourierFeature(n_features, feature_sampler, hyper_fixed = Dict("sigma" => s))
    return RandomFeatureMethod(sf, regularization = regularizer)
end


function calculate_mean_cov_and_coeffs(
    rng::AbstractRNG,
    l::Union{Real, AbstractVecOrMat},
    s::Real,
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Dict,
    io_pairs::PairedDataContainer,
)

    regularizer = noise_sd^2
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train

    # split data into train/test randomly
    itrain = get_inputs(io_pairs)[:, 1:n_train]
    otrain = get_outputs(io_pairs)[:, 1:n_train]
    io_train_cost = PairedDataContainer(itrain, otrain)
    itest = get_inputs(io_pairs)[:, (n_train + 1):end]
    otest = get_outputs(io_pairs)[:, (n_train + 1):end]
    input_dim = size(itrain, 1)

    # build and fit the RF
    rfm = RFM_from_hyperparameters(rng, l, s, regularizer, n_features, batch_sizes, input_dim)
    fitted_features = fit(rfm, io_train_cost, decomposition_type = "svd")

    test_batch_size = get_batch_size(rfm, "test")
    batch_inputs = batch_generator(itest, test_batch_size, dims = 2) # input_dim x batch_size

    #we want to calc 1/var(y-mean)^2 + lambda/m * coeffs^2 in the end
    pred_mean, pred_cov = predict(rfm, fitted_features, DataContainer(itest))
    scaled_coeffs = sqrt(1 / n_features) * get_coeffs(fitted_features)
    return pred_mean, pred_cov, scaled_coeffs

end


function estimate_mean_cov_and_coeffnorm_covariance(
    rng::AbstractRNG,
    l::Union{Real, AbstractVecOrMat},
    s::Real,
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Dict,
    io_pairs::PairedDataContainer,
    n_samples::Int;
    repeats::Int = 1,
)
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train
    means = zeros(n_test, n_samples)
    covs = zeros(n_test, n_samples)
    coeffl2norm = zeros(1, n_samples)
    for i in 1:n_samples
        for j in 1:repeats
            m, v, c = calculate_mean_cov_and_coeffs(rng, l, s, noise_sd, n_features, batch_sizes, io_pairs)
            means[:, i] += m' / repeats
            covs[:, i] += v' / repeats
            coeffl2norm[1, i] += sqrt(sum(c .^ 2)) / repeats
        end
    end
    #    println("take covariances")
    Γ = cov(vcat(means, covs, coeffl2norm), dims = 2)
    approx_σ2 = Diagonal(mean(covs, dims = 2)[:, 1]) # approx of \sigma^2I +rf var

    return Γ, approx_σ2

end

function calculate_ensemble_mean_cov_and_coeffnorm(
    rng::AbstractRNG,
    lvecormat::AbstractVecOrMat,
    s::Real,
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Dict,
    io_pairs::PairedDataContainer;
    repeats::Int = 1,
)
    if isa(lvecormat, AbstractVector)
        lmat = reshape(lvecormat, 1, :)
    else
        lmat = lvecormat
    end
    N_ens = size(lmat, 2)
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train

    means = zeros(n_test, N_ens)
    covs = zeros(n_test, N_ens)
    coeffl2norm = zeros(1, N_ens)
    for i in collect(1:N_ens)
        for j in collect(1:repeats)
            l = lmat[:, i]
            m, v, c = calculate_mean_cov_and_coeffs(rng, l, s, noise_sd, n_features, batch_sizes, io_pairs)
            means[:, i] += m' / repeats
            covs[:, i] += v' / repeats
            coeffl2norm[1, i] += sqrt(sum(c .^ 2)) / repeats
        end
    end
    return vcat(means, covs, coeffl2norm), Diagonal(mean(covs, dims = 2)[:, 1]) # approx of \sigma^2I 


end

## Begin Script, define problem setting
println("Begin script")
date_of_run = Date(2022, 9, 22)
input_dim_list = [6]

for input_dim in input_dim_list
    println("Number of input dimensions: ", input_dim)

    # radial
    # ftest_nd_to_1d(x::AbstractMatrix) = mapslices(column -> 1/(1+exp(-0.25*norm(column)^2)), x, dims=1)
    # not radial - different scale in each dimension
    ftest_nd_to_1d(x::AbstractMatrix) =
        mapslices(column -> exp(-0.1 * norm([i * c for (i, c) in enumerate(column)])^2), x, dims = 1)
    n_data = 20 * input_dim
    x = rand(rng, MvNormal(zeros(input_dim), 0.5 * I), n_data)
    noise_sd = 1e-3
    noise = rand(rng, Normal(0, noise_sd), (1, n_data))

    y = ftest_nd_to_1d(x) + noise
    io_pairs = PairedDataContainer(x, y)

    ## Define Hyperpriors for EKP

    μ_l = 5.0
    σ_l = 10.0
    # prior for non radial problem
    prior_lengthscale = constrained_gaussian("lengthscale", μ_l, σ_l, 0.0, Inf, repeats = input_dim)
    priors = prior_lengthscale

    μ_s = 1.0

    # estimate the noise from running many RFM sample costs at the mean values
    batch_sizes = Dict("train" => 600, "test" => 600, "feature" => 600)
    n_train = Int(floor(0.8 * n_data))
    n_test = n_data - n_train
    n_features = Int(floor(1.2 * n_data))
    # RF will perform poorly when n_features is close to n_train
    @assert(!(n_features == n_train)) #

    repeats = 1

    CALC_TRUTH = true

    println("RHKS norm type: norm of coefficients")

    if CALC_TRUTH
        sample_multiplier = 1

        n_samples = 2 * Int(floor(((1 + n_test) + 1) * sample_multiplier))
        println("Estimating output covariance with ", n_samples, " samples")
        internal_Γ, approx_σ2 = estimate_mean_cov_and_coeffnorm_covariance(
            rng,
            μ_l, # take mean values
            μ_s, # take mean values
            noise_sd,
            n_features,
            batch_sizes,
            io_pairs,
            n_samples,
            repeats = repeats,
        )


        save("calculated_truth_cov.jld2", "internal_Γ", internal_Γ)
    else
        println("Loading truth covariance from file...")
        internal_Γ = load("calculated_truth_cov.jld2")["internal_Γ"]
    end

    Γ = internal_Γ
    Γ[1:n_test, 1:n_test] += approx_σ2
    Γ[(n_test + 1):(2 * n_test), (n_test + 1):(2 * n_test)] += I
    Γ[(2 * n_test + 1):end, (2 * n_test + 1):end] += I

    println(
        "Estimated variance. Tr(cov) = ",
        tr(Γ[1:n_test, 1:n_test]),
        " + ",
        tr(Γ[(n_test + 1):(2 * n_test), (n_test + 1):(2 * n_test)]),
        " + ",
        tr(Γ[(2 * n_test + 1):end, (2 * n_test + 1):end]),
    )
    #println("noise in observations: ", Γ)
    # Create EKI
    N_ens = 10 * input_dim
    N_iter = 30
    update_cov_step = Inf

    initial_params = construct_initial_ensemble(priors, N_ens; rng_seed = ekp_seed)
    params_init = transform_unconstrained_to_constrained(priors, initial_params)[1, :]
    println("Prior gives parameters between: [$(minimum(params_init)),$(maximum(params_init))]")
    data = vcat(y[(n_train + 1):end], noise_sd^2 * ones(n_test), 0.0)

    ekiobj = [EKP.EnsembleKalmanProcess(initial_params, data, Γ, Inversion())]
    err = zeros(N_iter)
    println("Begin EKI iterations:")
    Δt = [1.0]

    for i in 1:N_iter

        #get parameters:
        lvec = transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1]))
        g_ens, _ = calculate_ensemble_mean_cov_and_coeffnorm(
            rng,
            lvec,
            μ_s,
            noise_sd,
            n_features,
            batch_sizes,
            io_pairs,
            repeats = repeats,
        )

        if i % update_cov_step == 0 # one update to the 

            constrained_u = transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1]))
            println("Estimating output covariance with ", n_samples, " samples")
            internal_Γ_new, approx_σ2_new = estimate_mean_cov_and_coeffnorm_covariance(
                rng,
                mean(constrained_u, dims = 2)[:, 1], # take mean values
                μ_s, # take mean values
                noise_sd,
                n_features,
                batch_sizes,
                io_pairs,
                n_samples,
                repeats = repeats,
            )
            Γ_new = internal_Γ_new
            Γ_new[1:n_test, 1:n_test] += approx_σ2_new
            Γ_new[(n_test + 1):(2 * n_test), (n_test + 1):(2 * n_test)] += I
            Γ_new[(2 * n_test + 1):end, (2 * n_test + 1):end] += I
            println(
                "Estimated variance. Tr(cov) = ",
                tr(Γ_new[1:n_test, 1:n_test]),
                " + ",
                tr(Γ_new[(n_test + 1):(2 * n_test), (n_test + 1):(2 * n_test)]),
                " + ",
                tr(Γ_new[(2 * n_test + 1):end, (2 * n_test + 1):end]),
            )

            ekiobj[1] = EKP.EnsembleKalmanProcess(get_u_final(ekiobj[1]), data, Γ_new, Inversion())

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
    n_data_test = 300 * input_dim
    n_features_test = Int(floor(1.2 * n_data_test))
    println("number of training data: ", n_data_test)
    println("number of features: ", n_features_test)
    x_test = rand(rng, MvNormal(zeros(input_dim), 0.5 * I), n_data_test)
    noise_test = rand(rng, Normal(0, noise_sd), (1, n_data_test))

    y_test = ftest_nd_to_1d(x_test) + noise_test
    io_pairs_test = PairedDataContainer(x_test, y_test)

    # get feature distribution
    final_lvec = mean(transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1])), dims = 2)
    println("**********")
    println("Optimal lengthscales: $(final_lvec)")
    println("**********")




    μ_c = 0.0
    if size(final_lvec, 1) == 1
        σ_c = repeat([final_lvec[1, 1]], input_dim)
    else
        σ_c = final_lvec[:, 1]
    end
    regularizer = noise_sd^2
    pd = ParameterDistribution(
        Dict(
            "distribution" => VectorOfParameterized(map(sd -> Normal(μ_c, sd), σ_c)),
            "constraint" => repeat([no_constraint()], input_dim),
            "name" => "xi",
        ),
    )
    feature_sampler = FeatureSampler(pd, rng = copy(rng))
    sigma_fixed = Dict("sigma" => 1.0)
    sff = ScalarFourierFeature(n_features_test, feature_sampler, hyper_fixed = sigma_fixed)

    #second case with batching

    rfm_batch = RandomFeatureMethod(sff, batch_sizes = batch_sizes, regularization = regularizer)
    fitted_batched_features = fit(rfm_batch, io_pairs_test, decomposition_type = "svd")

    if PLOT_FLAG
        #plot slice through one dimensions, others fixed to 0
        xrange = collect(-3:0.01:3)
        xslice = zeros(input_dim, length(xrange))
        figure_save_directory = joinpath(@__DIR__, "output", string(date_of_run))
        if !isdir(figure_save_directory)
            mkpath(figure_save_directory)
        end

        for direction in 1:input_dim
            xslicenew = copy(xslice)
            xslicenew[direction, :] = xrange

            yslice = ftest_nd_to_1d(xslicenew)

            pred_mean_slice, pred_cov_slice = predict(rfm_batch, fitted_batched_features, DataContainer(xslicenew))
            pred_cov_slice = max.(pred_cov_slice, 0.0)
            plt = plot(
                xrange,
                yslice',
                show = false,
                color = "black",
                linewidth = 5,
                size = (600, 600),
                legend = :topleft,
                label = "Target",
            )
            println(mean(2 * sqrt.(pred_cov_slice)))
            plot!(
                xrange,
                pred_mean_slice',
                ribbon = [2 * sqrt.(pred_cov_slice); 2 * sqrt.(pred_cov_slice)]',
                label = "Fourier",
                color = "blue",
            )
            savefig(
                plt,
                joinpath(
                    figure_save_directory,
                    "Fit_and_predict_ND_" * string(direction) * "of" * string(input_dim) * ".pdf",
                ),
            )
            savefig(
                plt,
                joinpath(
                    figure_save_directory,
                    "Fit_and_predict_ND_" * string(direction) * "of" * string(input_dim) * ".png",
                ),
            )

        end

    end
end
