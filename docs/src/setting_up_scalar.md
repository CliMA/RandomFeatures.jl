# Setting up a Scalar Random Feature Method

A basic creation of sigmoid-based scalar-valued random feature method is given as follows

```julia
# user inputs required:
# paired input-output data - io_pairs::PairedDataContainer 
# parameter distribution   - pd::ParameterDistribution 
# number of features       - n_features::Int

feature_sampler = FeatureSampler(pd) 
sigmoid_sf = ScalarFeature(n_features, feature_sampler, Sigmoid()) 
rfm = RandomFeatureMethod(sigmoid_sf)
fitted_features = fit(rfm, io_pairs)
```
Prediction at new inputs are made with
``` julia
# user inputs required
# new test inputs - i_test::DataContainer

predicted_mean, predicted_var = predict(rfm, fitted_features, i_test)
```
We see the core objects
- [`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/parameter_distributions/): a flexible container for constructing constrained parameter distributions, (from [`EnsembleKalmanProcesses.jl`](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/))
- [`(Paired)DataContainer`](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/internal_data_representation): consistent storage objects for input-output pairs or just inputs, (from [`EnsembleKalmanProcesses.jl`](https://clima.github.io/EnsembleKalmanProcesses.jl/stable))
- `FeatureSampler`: Builds the random feature distributions from a parameter distribution
- `ScalarFeature`: Builds a feature from the random feature distributions
- `RandomFeatureMethod`: Sets up the learning problem (with e.g. batching, regularization)
- `Fit`: Stores fitted features from the `fit` method

!!! note "See some examples!"
    Running the [test suite](@ref test-suite) with `TEST_PLOT_FLAG = true` will produce some ``1``-D``\to 1``-D and ``d``-D ``\to 1``-D example fits produced by [`test/Methods/runtests.jl`](https://github.com/CliMA/RandomFeatures.jl/tree/main/test/Methods). These use realistic optional arguments and distributions.

## `ParameterDistributions`

The simplest construction of parameter distributions is by using the `constrained_gaussian` construction.


```julia
using RandomFeatures.ParameterDistributions
```
#### **Recommended** univariate and product distribution build
The easiest constructors are for univariate and products
```julia
# constrained_gaussian("xi", desired_mean, desired_std, lower_bound, upper_bound)
one_dim_pd = constrained_gaussian("xi", 10, 5, -Inf, Inf) # Normal distribution
five_dim_pd = constrained_gaussian("xi", 10, 5, 0, Inf, repeats = 5) # Log-normal (approx mean 10 & approx std 5) in each of the five dimensions
```
#### Simple multivariate distribution
Simple unconstrained distribution is created as follows. 
```julia
using Distributions

μ = zeros(3)
Σ = SymTridiagonal(2 * ones(3), ones(2))
three_dim_pd = ParameterDistribution(
    Dict("distribution" => Parameterized(MvNormal(μ,Σ)), # the distribution
         "constraint" => repeat([no_constraint()],3), # constraints 
         "name" => "xi",
      ),
)
```
!!! note " xi? "
    The name of the distribution of the features **must** be `"xi"`
    
!!! note "Further distributions"
    Combined distributions can be made using the `VectorOfParameterized`, or histogram-based distributions with `Samples`. Extensive documentation of distributions and constraints is found [here](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/parameter_distributions/).

## `Sampler`
The random feature distribution ``\mathcal{D}`` is built of two distributions, the user-provided ``\mathcal{D}_\xi`` (`"xi"`) and a bias distribution (`"bias"`). The bias distribution is one-dimensional, and commonly uniformly distributed, so we provide an additional constructor for this case

```math
\theta = (\xi,b) \sim \mathcal{D} = (\mathcal{D}_\xi, \mathcal{U}([c_\ell,c_u]))
```
Defaults ``c_\ell = 0, c_u = 2\pi``. In the code this is built as
```julia
sampler = FeatureSampler(
    parameter_distribution;
    uniform_shift_bounds = [0, 2 * π], 
    rng = Random.GLOBAL_RNG
)
```
A random number generator can be provided. The second argument can be replaced with a general ``1``-D `ParameterDistribution` with a name-field `"bias"`.

## Features: `ScalarFeature` ``d``-D ``\to 1``-D

Given ``x\in\mathbb{R}^n`` input data, and ``m`` (`n_features`) features, `Features` produces samples of
```math
\Phi(x;\theta_j) = \sigma f(\xi_j\cdot x + b_j),\qquad \theta_j=(\xi_j,b_j) \sim \mathcal{D}\qquad \mathrm{for}\ j=1,\dots,m 
```
Note that ``\Phi \in \mathbb{R}^{n,m,p}``. Choosing``f`` as a cosine to produce fourier features
```julia
sf = ScalarFourierFeature(
    n_features,
    sampler;
    kwargs...
) 
```
``f`` as a neuron activation produces a neuron feature (`ScalarActivation` listed [here](@ref scalar-functions)) 
```julia
sf = ScalarNeuronFeature(
    n_features,
    sampler;
    activation_fun = Relu(),
    kwargs...
) 
```
The keyword `feature_parameters = Dict("sigma" => a)`, can be included to set the value of ``\sigma``.

## Method

The `RandomFeatureMethod` sets up the training problem to learn coefficients ``\beta\in\mathbb{R}^m`` from input-output training data ``(x,y)=\{(x_i,y_i)\}_{i=1}^n``, ``y_i \in \mathbb{R}`` and parameters ``\theta = \{\theta_j\}_{j=1}^m``:
```math
(\frac{1}{\lambda m}\Phi^T(x;\theta) \Phi(x;\theta) + I) \beta = \frac{1}{\lambda}\Phi^T(x;\theta)y
```
Where ``\lambda>0`` is a regularization parameter that effects the `+I`. The equation is written in this way to be consistent with the vector-output case.
```julia
rfm = RandomFeatureMethod(
    sf;
    regularization = 1e12 * eps() * I,
    batch_sizes = ("train" => 0, "test" => 0, "feature" => 0),
)
```
One can select batch sizes to balance the space-time (memory-process) trade-off. when building and solving equations by setting values of `"train"`, test data `"test"` and number of features `"feature"`. The default is no batching (`0`).

!!! warning "Conditioning"
    The problem is ill-conditioned without regularization.
    If you encounters a Singular or Positive-definite exceptions, try increasing `regularization`

The solve for ``\beta`` occurs in the `fit` method. In the `Fit` object we store the system matrix, its factorization, and the inverse of the factorization. For many applications this is most efficient representation, as predictions (particularly of the covariance) are then only a matrix multiplication.
```julia
fitted_features = fit(
    rfm,
    io_pairs; # (x,y)
    decomposition = "cholesky",
)
```
The decomposition is based off the [`LinearAlgebra.Factorize`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#man-linalg-factorizations) functions. For performance we have implemented only ("cholesky", "svd", and for ``\Lambda=0`` (not recommended) the method defaults to "pinv")

## Hyperparameters

[Coming soon]

!!! note
    Hyperparameter selection is very important for a good random feature fit.

The hyperparameters are the parameters appearing in the random feature distribution ``\mathcal{D}``. We have an examples where an ensemble-based algorithm is used to optimize such parameters in [`examples/Learn_hyperparameters/`](https://github.com/CliMA/RandomFeatures.jl/tree/main/examples/Learn_hyperparameters)


