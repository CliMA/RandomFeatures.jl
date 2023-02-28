module Utilities

using LinearAlgebra, DocStringExtensions, Tullio

export batch_generator,
    Decomposition, StoredInvType, Factor, PseInv, get_decomposition, get_full_matrix, get_parametric_type, linear_solve

"""
$(TYPEDSIGNATURES)

produces batched sub-array views of size `batch_size` along dimension `dims`.
!!! note
    this creates views not copies. Modifying a batch will modify the original!

"""
function batch_generator(array::AbstractArray, batch_size::Int; dims::Int = 1)
    if batch_size == 0
        return [array]
    end

    n_batches = Int(ceil(size(array, dims) / batch_size))
    batch_idx = [
        i < n_batches ? collect(((i - 1) * batch_size + 1):(i * batch_size)) :
        collect(((i - 1) * batch_size + 1):size(array, dims)) for i in 1:n_batches
    ]

    return [selectdim(array, dims, b) for b in batch_idx]
end


# Decomposition/linear solves for feature matrices

"""
$(TYPEDEF)

Type used as a flag for the stored Decomposition type
"""
abstract type StoredInvType end

"""
$(TYPEDEF)
"""
abstract type Factor <: StoredInvType end

"""
$(TYPEDEF)
"""
abstract type PseInv <: StoredInvType end

"""
$(TYPEDEF)

Stores a matrix along with a decomposition `T=Factor`, or pseudoinverse `T=PseInv`

$(TYPEDFIELDS)
"""
struct Decomposition{T}
    "The original matrix"
    full_matrix::AbstractMatrix
    "The matrix decomposition, or pseudoinverse"
    decomposition::Union{AbstractMatrix, Factorization}
end

function Decomposition(mat::AbstractMatrix, method::AbstractString; nugget::Real = 1e12 * eps())
    if method == "pinv"
        decomposition = pinv(mat)
        return Decomposition{PseInv}(mat, decomposition)

    else
        if !isdefined(LinearAlgebra, Symbol(method))
            throw(
                ArgumentError(
                    "factorization method " *
                    string(method) *
                    " not found in LinearAlgebra, please select one of the existing options",
                ),
            )
        else
            f = getfield(LinearAlgebra, Symbol(method))

            if method == "cholesky"
                if !isposdef(mat)
                    @info "Random Feature system not positive definite. Performing cholesky factorization with a close positive definite matrix"
                    mat = 0.5 * (mat + permutedims(mat, (2, 1))) #symmetrize
                    correction = abs(minimum(eigvals(mat))) + nugget #add min eval + eps
                    mat += correction * I
                end

            end
            decomposition = f(mat)
            return Decomposition{Factor}(mat, decomposition)
        end
    end
end
"""
$(TYPEDSIGNATURES)

get `decomposition` field
"""
get_decomposition(d::Decomposition) = d.decomposition

"""
$(TYPEDSIGNATURES)

get `full_matrix` field
"""
get_full_matrix(d::Decomposition) = d.full_matrix

"""
$(TYPEDSIGNATURES)

get the parametric type
"""
get_parametric_type(d::Decomposition{T}) where {T} = T

"""
$(TYPEDSIGNATURES)

Solve the linear system based on `Decomposition` type
"""
function linear_solve(d::Decomposition, rhs::AbstractArray, ::Type{Factor})
    #    return get_decomposition(d) \ rhs
    M, N, P = size(rhs)
    x = zeros(M, N, P)
    for p in 1:P # \ can handle "Matrix \ Matrix", but not "Matrix \ 3-tensor"
        x[:, :, p] = get_decomposition(d) \ rhs[:, :, p]
    end
    return x
end
function linear_solve(d::Decomposition, rhs::AbstractArray, ::Type{PseInv})
    #get_decomposition(d) * rhs
    @tullio x[m, n, p] := get_decomposition(d)[m, i] * rhs[i, n, p]
    return x
end

linear_solve(d::Decomposition, rhs::AbstractArray) =
    linear_solve(d::Decomposition, rhs::AbstractArray, get_parametric_type(d))

end
