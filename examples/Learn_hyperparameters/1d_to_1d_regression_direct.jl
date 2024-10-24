# Example to learn hyperparameters of simple 1d-1d regression example.
# This example matches test/Methods/runtests.jl testset: "Fit and predict: 1D -> 1D"
# The (approximate) optimal values here are used in those tests.


using StableRNGs
using Distributions
using StatsBase
using LinearAlgebra
using Random
using Dates


PLOT_FLAG = true
println("plot flag: ", PLOT_FLAG)
if PLOT_FLAG
    using Plots, ColorSchemes
end

using EnsembleKalmanProcesses
const EKP = EnsembleKalmanProcesses

using RandomFeatures.ParameterDistributions
using RandomFeatures.DataContainers
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
    l::Real,
    regularizer::Real,
    n_features::Int,
    batch_sizes::Dict,
    feature_type::String,
)

    μ_c = 0.0
    σ_c = l
    pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
    feature_sampler = FeatureSampler(pd, rng = rng)
    # Learn hyperparameters for different feature types

    if feature_type == "fourier"
        sf = ScalarFourierFeature(n_features, feature_sampler)
    elseif feature_type == "neuron"
        sf = ScalarNeuronFeature(n_features, feature_sampler)
    elseif feature_type == "sigmoid"
        sf = ScalarFeature(n_features, feature_sampler, Sigmoid())
    end
    return RandomFeatureMethod(sf, batch_sizes = batch_sizes, regularization = regularizer)
end


function calculate_mean_and_coeffs(
    rng::AbstractRNG,
    l::Real,
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Dict,
    io_pairs::PairedDataContainer,
    feature_type::String,
)
    regularizer = noise_sd^2
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train

    # split data into train/test randomly
    itrain = reshape(get_inputs(io_pairs)[1, 1:n_train], 1, :)
    otrain = reshape(get_outputs(io_pairs)[1, 1:n_train], 1, :)
    io_train_cost = PairedDataContainer(itrain, otrain)
    itest = reshape(get_inputs(io_pairs)[1, (n_train + 1):end], 1, :)
    otest = reshape(get_outputs(io_pairs)[1, (n_train + 1):end], 1, :)

    # build and fit the RF
    rfm = RFM_from_hyperparameters(rng, l, regularizer, n_features, batch_sizes, feature_type)
    fitted_features = fit(rfm, io_train_cost)

    test_batch_size = get_batch_size(rfm, "test")
    batch_inputs = batch_generator(itest, test_batch_size, dims = 2) # input_dim x batch_size

    #we want to calc lambda/m * coeffs^2 in the end
    pred_mean, features = predictive_mean(rfm, fitted_features, DataContainer(itest))
    scaled_coeffs = sqrt(1 / n_features) * get_coeffs(fitted_features)
    chol_fac = get_decomposition(get_feature_factors(fitted_features)).L
    complexity = 2 * sum(log(chol_fac[i, i]) for i in 1:size(chol_fac, 1))
    complexity = sqrt(complexity)
    return pred_mean, scaled_coeffs, complexity

end

function estimate_mean_and_coeffnorm_covariance(
    rng::AbstractRNG,
    l::Real,
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Dict,
    io_pairs::PairedDataContainer,
    feature_type::String,
    n_samples::Int;
    repeats::Int = 1,
)
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train
    means = zeros(n_test, n_samples)
    coeffl2norm = zeros(1, n_samples)
    complexity = zeros(1, n_samples)

    for i in 1:n_samples
        for j in 1:repeats
            m, c, cplx = calculate_mean_and_coeffs(rng, l, noise_sd, n_features, batch_sizes, io_pairs, feature_type)
            means[:, i] += m' / repeats
            coeffl2norm[1, i] += sqrt(sum(c .^ 2)) / repeats
            complexity[1, i] += cplx / repeats

        end
    end

    Γ = cov(vcat(means, coeffl2norm, complexity), dims = 2)
    return Γ

end

function calculate_ensemble_mean_and_coeffnorm(
    rng::AbstractRNG,
    lvec::AbstractVector,
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Dict,
    io_pairs::PairedDataContainer,
    feature_type::String;
    repeats::Int = 1,
)
    N_ens = length(lvec)
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train

    means = zeros(n_test, N_ens)
    coeffl2norm = zeros(1, N_ens)
    complexity = zeros(1, N_ens)
    for (i, l) in zip(collect(1:N_ens), lvec)
        for j in collect(1:repeats)
            m, c, cplx = calculate_mean_and_coeffs(rng, l, noise_sd, n_features, batch_sizes, io_pairs, feature_type)
            means[:, i] += m' / repeats
            coeffl2norm[1, i] += sqrt(sum(c .^ 2)) / repeats
            complexity[1, i] += cplx / repeats
        end
    end
    return vcat(means, coeffl2norm, complexity)

end



## Begin Script, define problem setting

println("starting script")
date_of_run = Date(2024, 4, 10)


# Target function
ftest(x::AbstractVecOrMat) = exp.(-0.5 * x .^ 2) .* (x .^ 4 - x .^ 3 - x .^ 2 + x .- 1)

n_data = 20 * 2
noise_sd = 0.1

x = rand(rng, Uniform(-3, 3), n_data)
noise = rand(rng, Normal(0, noise_sd), n_data)
y = ftest(x) + noise

io_pairs = PairedDataContainer(reshape(x, 1, :), reshape(y, 1, :), data_are_columns = true) #matrix input

xtestvec = collect(-3:0.01:3)
ntest = length(xtestvec) #extended domain
xtest = DataContainer(reshape(xtestvec, 1, :), data_are_columns = true)
ytest = ftest(get_data(xtest))

## Define Hyperpriors for EKP

μ_l = 10.0
σ_l = 10.0
prior_lengthscale = constrained_gaussian("lengthscale", μ_l, σ_l, 0.0, Inf)

priors = prior_lengthscale

# estimate the noise from running many RFM sample costs at the mean values
batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)
n_train = Int(floor(0.8 * n_data))
n_test = n_data - n_train
n_samples = n_test + 1 # >  n_test
n_features = 300
repeats = 1


feature_types = ["fourier", "neuron", "sigmoid"]
lengthscales = zeros(length(feature_types))
for (idx, type) in enumerate(feature_types)
    println("estimating noise in observations... ")
    internal_Γ = estimate_mean_and_coeffnorm_covariance(
        rng,
        μ_l, # take mean values
        noise_sd,
        n_features,
        batch_sizes,
        io_pairs,
        type,
        n_samples,
        repeats = repeats,
    )
    Γ = internal_Γ
    Γ[1:n_test, 1:n_test] += noise_sd^2 * I
    Γ[(n_test + 1):end, (n_test + 1):end] += I


    println("Finished estimation.")
    # Create EKI
    N_ens = 50
    N_iter = 20
    initial_params = construct_initial_ensemble(rng, priors, N_ens)
    data = vcat(y[(n_train + 1):end], 0.0, 0.0)
    ekiobj = [
        EKP.EnsembleKalmanProcess(
            initial_params,
            data,
            Γ,
            Inversion(),
            scheduler = DataMisfitController(terminate_at = 1000),
        ),
    ]


    err = zeros(N_iter)
    for i in 1:N_iter

        #get parameters:
        constrained_u = transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1]))
        lvec = constrained_u[1, :]
        g_ens = calculate_ensemble_mean_and_coeffnorm(
            rng,
            lvec,
            noise_sd,
            n_features,
            batch_sizes,
            io_pairs,
            type,
            repeats = repeats,
        )
        terminated = EKP.update_ensemble!(ekiobj[1], g_ens)
        if !isnothing(terminated)
            break
        end

        err[i] = get_error(ekiobj[1])[end] #mean((params_true - mean(params_i,dims=2)).^2)
        println(
            "Iteration: " *
            string(i) *
            ", Error: " *
            string(err[i]) *
            ", with parameter mean" *
            string(mean(transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1])), dims = 2)[:, 1]),
            " and sd ",
            string(sqrt.(var(transform_unconstrained_to_constrained(priors, get_u_final(ekiobj[1])), dims = 2))[:, 1]),
        )

    end

    lengthscales[idx] = transform_unconstrained_to_constrained(priors, mean(get_u_final(ekiobj[1]), dims = 2))[1, 1]
end


println("****")
println("Optimal lengthscales:  ", feature_types, " = ", lengthscales)
println("****")

#run an actual experiment
n_features_test = 1000
n_data_test = 300
x_test = rand(rng, Uniform(-3, 3), (1, n_data_test))
noise_test = rand(rng, Normal(0, noise_sd), (1, n_data_test))

y_test = ftest(x_test) + noise_test
io_pairs_test = PairedDataContainer(x_test, y_test)

μ_c = 0.0
σ_c = lengthscales

regularizer = noise_sd^2

rfms = Any[]
fits = Any[]
for (idx, sd, feature_type) in zip(collect(1:length(σ_c)), σ_c, feature_types)
    pd = constrained_gaussian("xi", 0.0, sd, -Inf, Inf)

    feature_sampler = FeatureSampler(pd, rng = copy(rng))

    if feature_type == "fourier"
        sf = ScalarFourierFeature(n_features_test, feature_sampler)
    elseif feature_type == "neuron"
        sf = ScalarNeuronFeature(n_features_test, feature_sampler)
    elseif feature_type == "sigmoid"
        sf = ScalarFeature(n_features_test, feature_sampler, Sigmoid())
    end

    push!(rfms, RandomFeatureMethod(sf, batch_sizes = batch_sizes, regularization = regularizer))
    push!(fits, fit(rfms[end], io_pairs_test))
end

if PLOT_FLAG
    figure_save_directory = joinpath(@__DIR__, "output", string(date_of_run))
    if !isdir(figure_save_directory)
        mkpath(figure_save_directory)
    end
    #plot slice through one dimensions, others fixed to 0
    xplt = reshape(collect(-3:0.01:3), 1, :)
    yplt = ftest(xplt)
    clrs = map(x -> get(colorschemes[:hawaii], x), [0.25, 0.5, 0.75])

    plt = plot(
        xplt',
        yplt',
        show = false,
        color = "black",
        linewidth = 5,
        size = (600, 600),
        legend = :topleft,
        label = "Target",
    )
    for (idx, rfm, fit, feature_type, clr) in zip(collect(1:length(σ_c)), rfms, fits, feature_types, clrs)

        pred_mean, pred_cov = predict(rfm, fit, DataContainer(xplt))
        pred_cov = pred_cov[1, 1, :] #just variances
        pred_cov = max.(pred_cov, 0.0) #not normally needed..
        plot!(
            xplt',
            pred_mean',
            ribbon = [2 * sqrt.(pred_cov); 2 * sqrt.(pred_cov)]',
            label = feature_type,
            color = clr,
        )

    end
    savefig(plt, joinpath(figure_save_directory, "Fit_and_predict_1D.pdf"))

end
