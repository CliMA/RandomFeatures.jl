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

Type of a function mapping 1D -> 1D
"""
abstract type ScalarFunction end

"""
$(TYPEDSIGNATURES)

apply the scalar function `sf` pointwise to vectors or matrices
"""
apply_scalar_function(sf::SF, r::A) where {SF <: ScalarFunction, A <: AbstractArray} =
    apply_scalar_function.(Ref(sf), r) # Ref(sf) treats sf as a scalar for the broadcasting

"""
$(TYPEDEF)
"""
struct Cosine <: ScalarFunction end
function apply_scalar_function(sf::Cosine, r::FT) where {FT <: AbstractFloat}
    return cos(r)
end

# specific set used for neurons
"""
$(TYPEDEF)

Type of scalar activation functions
"""
abstract type ScalarActivation <: ScalarFunction end

"""
$(TYPEDEF)
"""
struct Relu <: ScalarActivation end
function apply_scalar_function(sa::Relu, r::FT) where {FT <: AbstractFloat}
    return max(0, r)
end

"""
$(TYPEDEF)
"""
struct Gelu <: ScalarActivation end
function apply_scalar_function(sa::Gelu, r::FT) where {FT <: AbstractFloat}
    cdf = 0.5 * (1.0 + erf(r / sqrt(2.0)))
    return r * cdf
end

"""
$(TYPEDEF)
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
"""
struct Sawtooth <: ScalarActivation end
function apply_scalar_function(sa::Sawtooth, r::FT) where {FT <: AbstractFloat}
    return max(0, min(2 * r, 2 - 2 * r))
end

"""
$(TYPEDEF)
"""
struct Softplus <: ScalarActivation end
function apply_scalar_function(sa::Softplus, r::FT) where {FT <: AbstractFloat}
    return log(1 + exp(-abs(r))) + max(r, 0)
end

"""
$(TYPEDEF)
"""
struct Tansig <: ScalarActivation end
function apply_scalar_function(sa::Tansig, r::FT) where {FT <: AbstractFloat}
    return tanh(r)
end

"""
$(TYPEDEF)
"""
struct Sigmoid <: ScalarActivation end
function apply_scalar_function(sa::Sigmoid, r::FT) where {FT <: AbstractFloat}
    return 1 / (1 + exp(-r))
end

"""
$(TYPEDEF)
"""
@kwdef struct Elu{FT <: AbstractFloat} <: ScalarActivation
    alpha::FT = 1.0
end
function apply_scalar_function(sa::Elu, r::FT) where {FT <: AbstractFloat}
    return r > 0 ? r : sa.alpha * (exp(r) - 1.0)
end

"""
$(TYPEDEF)
"""
@kwdef struct Lrelu{FT <: AbstractFloat} <: ScalarActivation
    alpha::FT = 0.01
end
function apply_scalar_function(sa::Lrelu, r::FT) where {FT <: AbstractFloat}
    return r > 0 ? r : sa.alpha * r
end

"""
$(TYPEDEF)
"""
@kwdef struct Selu{FT <: AbstractFloat} <: ScalarActivation
    alpha::FT = 1.67326
    lambda::FT = 1.0507
end
function apply_scalar_function(sa::Selu, r::FT) where {FT <: AbstractFloat}
    return r > 0 ? sa.lambda * r : sa.lambda * sa.alpha * (exp(r) - 1.0)
end

"""
$(TYPEDEF)
"""
@kwdef struct SmoothHeaviside{FT <: AbstractFloat} <: ScalarActivation
    epsilon::FT = 0.01
end
function apply_scalar_function(sa::SmoothHeaviside, r::FT) where {FT <: AbstractFloat}
    return 1 / 2 + (1 / pi) * atan(r / sa.epsilon)
end
