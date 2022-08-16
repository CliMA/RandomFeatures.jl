# list of scalar function - here usage is e.g.,
# SA = Relu()
# apply_scalar_function(SA,r)  

using SpecialFunctions
import Base.@kwdef

abstract type ScalarFunction end

struct Cosine <: ScalarActivation end
function apply_scalar_function(sa::Cosine,r)
    return cos(r)
end

# specific set used for neurons

abstract type ScalarActivation <: ScalarFunction end

struct Relu <: ScalarActivation end
function apply_scalar_function(sa::Relu,r)
    return max(0,r)
end

struct Gelu <: ScalarActivation end
function apply_scalar_function(sa::Gelu,r)
    cdf = 0.5 * (1.0 + erf(x / sqrt(2.0)))
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

function apply_scalar_function(sa::Heaviside,r)
    return heaviside(r,0.5)
end

struct Sawtooth <: ScalarActivation end
function apply_scalar_function(sa::Sawtooth,r)
    return max(0,min(2 * r, 2 - 2 * r))
end

struct Softplus <: ScalarActivation end
function apply_scalar_function(sa::Softplus,r)
    return log(1 + exp(-abs(r))) + max(r,0)
end

struct Tansig <: ScalarActivation end
function apply_scalar_function(sa::Tansig,r)
    return tanh(r)
end

struct Sigmoid <: ScalarActivation end
function apply_scalar_function(sa::Sigmoid,r)
    return 1 / (1 + exp(-r))
end

@kwdef struct Elu <: ScalarActivation
    alpha::Real = 1.0
end
function apply_scalar_function(sa::Elu,r) 
    return r > 0 ? r : sa.alpha * (exp(r) - 1.0)
end

@kwdef struct Lrelu <: ScalarActivation
    alpha::Real = 0.01
end
function apply_scalar_function(sa::Lrelu,r)
    return r > 0 ? r : sa.alpha * r
end

@kwdef struct Selu <: ScalarActivation
    alpha::Real = 1.67326
    lambda::Real = 1.0507
end
function apply_scalar_function(sa::Selu,r)
    return r > 0 ? sa.lambda*r : sa.lambda*sa.alpha * (exp(r) - 1.0)
end

@kwdef struct SmoothHeaviside <: ScalarActivation
    epsilon::Real = 0.01
end
function apply_scalar_function(sa::SmoothHeaviside,r)
    return 1 / 2 + (1 / pi)*atan(r / sa.epsilon)
end












