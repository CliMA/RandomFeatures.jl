# list of scalar function - here usage is e.g.,
# SA = Relu()
# apply_scalar_function(SA,r)  

using SpecialFunctions
using DocStringExtensions
import Base.@kwdef


export ScalarFunction,
    ScalarActivation,
    Cosine,
    Relu,
    Lrelu,
    Gelu,
    Elu,
    Selu,
    Heaviside,
    SmoothHeaviside,
    Sawtooth,
    Softplus,
    Tansig,
    Sigmoid

export apply_scalar_function

"""
$(TYPEDEF)

Abstract supertype for all scalar functions mapping a single real value to a single real value.
"""
abstract type ScalarFunction end

"""
$(TYPEDSIGNATURES)

Apply the scalar function `sf` pointwise to each element of `r`.

# Examples
```jldoctest
julia> using RandomFeatures.Features

julia> apply_scalar_function(Relu(), [-1.0, 0.0, 1.0])
3-element Vector{Float64}:
 0.0
 0.0
 1.0
```
"""
apply_scalar_function(sf::SF, r::A) where {SF <: ScalarFunction, A <: AbstractArray} =
    apply_scalar_function.(Ref(sf), r) # Ref(sf) treats sf as a scalar for the broadcasting

"""
$(TYPEDEF)

Cosine scalar function: applies `cos(r)` pointwise to each input.
"""
struct Cosine <: ScalarFunction end
function apply_scalar_function(sf::Cosine, r::FT) where {FT <: AbstractFloat}
    return cos(r)
end

# specific set used for neurons
"""
$(TYPEDEF)

Abstract supertype for neural-network activation functions (a subtype of `ScalarFunction`).
"""
abstract type ScalarActivation <: ScalarFunction end

"""
$(TYPEDEF)

Rectified linear unit (ReLU) activation: returns `max(0, r)` for each input.
"""
struct Relu <: ScalarActivation end
function apply_scalar_function(sa::Relu, r::FT) where {FT <: AbstractFloat}
    return max(0, r)
end

"""
$(TYPEDEF)

Gaussian error linear unit (GELU) activation: `r * Φ(r)`, where `Φ` is the standard normal CDF.
"""
struct Gelu <: ScalarActivation end
function apply_scalar_function(sa::Gelu, r::FT) where {FT <: AbstractFloat}
    cdf = 0.5 * (1.0 + erf(r / sqrt(2.0)))
    return r * cdf
end

"""
$(TYPEDEF)

Heaviside step function: returns `0` for `r < 0`, `0.5` at `r == 0`, and `1` for `r > 0`.
"""
struct Heaviside <: ScalarActivation end
function heaviside(x, y)
    if x < 0
        return 0
    elseif x == 0
        return y
    else
        return 1
    end
end

function apply_scalar_function(sa::Heaviside, r::FT) where {FT <: AbstractFloat}
    return heaviside(r, 0.5)
end

"""
$(TYPEDEF)

Sawtooth (triangle wave) activation: `max(0, min(2r, 2 - 2r))`.
"""
struct Sawtooth <: ScalarActivation end
function apply_scalar_function(sa::Sawtooth, r::FT) where {FT <: AbstractFloat}
    return max(0, min(2 * r, 2 - 2 * r))
end

"""
$(TYPEDEF)

Softplus activation: numerically stable smooth approximation to ReLU, computed as `log(1 + exp(-|r|)) + max(r, 0)`.
"""
struct Softplus <: ScalarActivation end
function apply_scalar_function(sa::Softplus, r::FT) where {FT <: AbstractFloat}
    return log(1 + exp(-abs(r))) + max(r, 0)
end

"""
$(TYPEDEF)

Hyperbolic tangent sigmoid activation: returns `tanh(r)`.
"""
struct Tansig <: ScalarActivation end
function apply_scalar_function(sa::Tansig, r::FT) where {FT <: AbstractFloat}
    return tanh(r)
end

"""
$(TYPEDEF)

Logistic sigmoid activation: returns `1 / (1 + exp(-r))`.
"""
struct Sigmoid <: ScalarActivation end
function apply_scalar_function(sa::Sigmoid, r::FT) where {FT <: AbstractFloat}
    return 1 / (1 + exp(-r))
end

"""
$(TYPEDEF)

Exponential linear unit (ELU) activation: linear for `r > 0`, scaled exponential `alpha * (exp(r) - 1)` for `r ≤ 0`.

$(TYPEDFIELDS)

# Constructors

`Elu(; alpha = 1.0)`
"""
@kwdef struct Elu{FT <: AbstractFloat} <: ScalarActivation
    "Scale applied to the negative exponential branch [dimensionless]; default `1.0`."
    alpha::FT = 1.0
end
function apply_scalar_function(sa::Elu, r::FT) where {FT <: AbstractFloat}
    return r > 0 ? r : sa.alpha * (exp(r) - 1.0)
end

"""
$(TYPEDEF)

Leaky ReLU activation: linear for `r > 0`, scaled by `alpha` for `r ≤ 0`.

$(TYPEDFIELDS)

# Constructors

`Lrelu(; alpha = 0.01)`
"""
@kwdef struct Lrelu{FT <: AbstractFloat} <: ScalarActivation
    "Slope of the negative linear branch [dimensionless]; default `0.01`."
    alpha::FT = 0.01
end
function apply_scalar_function(sa::Lrelu, r::FT) where {FT <: AbstractFloat}
    return r > 0 ? r : sa.alpha * r
end

"""
$(TYPEDEF)

Scaled ELU (SELU) activation: self-normalising variant of ELU with fixed default scale parameters.

$(TYPEDFIELDS)

# Constructors

`Selu(; alpha = 1.67326, lambda = 1.0507)`
"""
@kwdef struct Selu{FT <: AbstractFloat} <: ScalarActivation
    "Negative-branch scale factor for self-normalisation [dimensionless]; default `1.67326`."
    alpha::FT = 1.67326
    "Global scale factor for self-normalisation [dimensionless]; default `1.0507`."
    lambda::FT = 1.0507
end
function apply_scalar_function(sa::Selu, r::FT) where {FT <: AbstractFloat}
    return r > 0 ? sa.lambda * r : sa.lambda * sa.alpha * (exp(r) - 1.0)
end

"""
$(TYPEDEF)

Smooth differentiable approximation to the Heaviside step function: `0.5 + atan(r / epsilon) / π`.

$(TYPEDFIELDS)

# Constructors

`SmoothHeaviside(; epsilon = 0.01)`
"""
@kwdef struct SmoothHeaviside{FT <: AbstractFloat} <: ScalarActivation
    "Width of the transition region [dimensionless]; smaller values approach the hard step function. Default `0.01`."
    epsilon::FT = 0.01
end
function apply_scalar_function(sa::SmoothHeaviside, r::FT) where {FT <: AbstractFloat}
    return 1 / 2 + (1 / pi) * atan(r / sa.epsilon)
end
