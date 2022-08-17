# list of scalar function - here usage is e.g.,
# SA = Relu()
# apply_scalar_function(SA,r)  

using SpecialFunctions
import Base.@kwdef

export
    ScalarFunction,
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

export
    apply_scalar_function


abstract type ScalarFunction end

apply_scalar_function(sf::ScalarFunction, r::AbstractVecOrMat) = apply_scalar_function.(Ref(sf),r) # Ref(sf) treats sf as a scalar for the broadcasting

struct Cosine <: ScalarFunction end
function apply_scalar_function(sf::Cosine, r::Real)
    return cos(r)
end

# specific set used for neurons

abstract type ScalarActivation <: ScalarFunction end

struct Relu <: ScalarActivation end
function apply_scalar_function(sa::Relu,r::Real)
    return max(0,r)
end

struct Gelu <: ScalarActivation end
function apply_scalar_function(sa::Gelu,r::Real)
    cdf = 0.5 * (1.0 + erf(r / sqrt(2.0)))
    return r * cdf
end

struct Heaviside <: ScalarActivation end
function heaviside(x,y)
    if x < 0
        return 0
    elseif x == 0
        return y
    else
        return 1
    end 
end

function apply_scalar_function(sa::Heaviside,r::Real)
    return heaviside(r,0.5)
end

struct Sawtooth <: ScalarActivation end
function apply_scalar_function(sa::Sawtooth,r::Real)
    return max(0,min(2 * r, 2 - 2 * r))
end

struct Softplus <: ScalarActivation end
function apply_scalar_function(sa::Softplus,r::Real)
    return log(1 + exp(-abs(r))) + max(r,0)
end

struct Tansig <: ScalarActivation end
function apply_scalar_function(sa::Tansig,r::Real)
    return tanh(r)
end

struct Sigmoid <: ScalarActivation end
function apply_scalar_function(sa::Sigmoid,r::Real)
    return 1 / (1 + exp(-r))
end

@kwdef struct Elu <: ScalarActivation
    alpha::Real = 1.0
end
function apply_scalar_function(sa::Elu,r::Real) 
    return r > 0 ? r : sa.alpha * (exp(r) - 1.0)
end

@kwdef struct Lrelu <: ScalarActivation
    alpha::Real = 0.01
end
function apply_scalar_function(sa::Lrelu,r::Real)
    return r > 0 ? r : sa.alpha * r
end

@kwdef struct Selu <: ScalarActivation
    alpha::Real = 1.67326
    lambda::Real = 1.0507
end
function apply_scalar_function(sa::Selu,r::Real)
    return r > 0 ? sa.lambda*r : sa.lambda*sa.alpha * (exp(r) - 1.0)
end

@kwdef struct SmoothHeaviside <: ScalarActivation
    epsilon::Real = 0.01
end
function apply_scalar_function(sa::SmoothHeaviside,r::Real)
    return 1 / 2 + (1 / pi)*atan(r / sa.epsilon)
end












