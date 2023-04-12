module Utilities

using LinearAlgebra, DocStringExtensions, Tullio

export batch_generator,
    Decomposition,
    StoredInvType,
    Factor,
    PseInv,
    get_decomposition,
    get_full_matrix,
    get_parametric_type,
    linear_solve,
    posdef_correct


"""
$(TYPEDSIGNATURES)

produces batched sub-array views of size `batch_size` along dimension `dims`.
!!! note
    this creates views not copies. Modifying a batch will modify the original!

"""
function batch_generator(array::A, batch_size::Int; dims::Int = 1) where {A <: AbstractArray}
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
    posdef_correct(mat::AbstractMatrix; tol::Real=1e8*eps())

Makes square matrix `mat` positive definite, by symmetrizing and bounding the minimum eigenvalue below by `tol`
"""
function posdef_correct(mat::M; tol::Real = 1e12 * eps()) where {M <: AbstractMatrix}
    out = 0.5 * (mat + permutedims(mat, (2, 1))) #symmetrize
    out += (abs(minimum(eigvals(out))) + tol) * I #add to diag
    return out
end


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
struct Decomposition{T, M <: AbstractMatrix, MorF <: Union{AbstractMatrix, Factorization}}
    "The original matrix"
    full_matrix::M
    "The matrix decomposition, or pseudoinverse"
    decomposition::MorF
end

function Decomposition(
    mat::M,
    method::S;
    nugget::R = 1e12 * eps(),
) where {M <: AbstractMatrix, S <: AbstractString, R <: Real}
    if method == "pinv"
        return Decomposition{PseInv, M, M}(mat, pinv(mat))
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
            # Don't use in-place, as we need full mat later too
            #            if isdefined(LinearAlgebra, Symbol(method*"!"))
            #                f = getfield(LinearAlgebra, Symbol(method*"!"))
            #            else
            f = getfield(LinearAlgebra, Symbol(method))
            #            end

            if method == "cholesky"
                if !isposdef(mat)
                    @info "Random Feature system not positive definite. Performing cholesky factorization with a close positive definite matrix"
                    mat = posdef_correct(mat, tol = nugget)
                end

            end

            return Decomposition{Factor, typeof(mat), Base.return_types(f, (typeof(mat),))[1]}(mat, f(mat))
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
get_parametric_type(d::Decomposition{T, M}) where {T, M <: Union{AbstractMatrix, Factorization}} = T

"""
$(TYPEDSIGNATURES)

Solve the linear system based on `Decomposition` type
"""
function linear_solve(d::Decomposition, rhs::A, ::Type{Factor}) where {A <: AbstractArray}
    #    return get_decomposition(d) \ rhs
    M, N, P = size(rhs)
    x = zeros(M, N, P)
    for p in 1:P # \ can handle "Matrix \ Matrix", but not "Matrix \ 3-tensor"
        x[:, :, p] = get_decomposition(d) \ rhs[:, :, p]
    end
    return x
end
function linear_solve(d::Decomposition, rhs::A, ::Type{PseInv}) where {A <: AbstractArray}
    #get_decomposition(d) * rhs
    @tullio x[m, n, p] := get_decomposition(d)[m, i] * rhs[i, n, p]
    return x
end

linear_solve(d::Decomposition, rhs::A) where {A <: AbstractArray} = linear_solve(d, rhs, get_parametric_type(d))

end
