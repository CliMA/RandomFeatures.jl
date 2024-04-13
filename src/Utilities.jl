module Utilities

using LinearAlgebra, DocStringExtensions, Tullio, LoopVectorization

export batch_generator,
    Decomposition,
    StoredInvType,
    Factor,
    PseInv,
    get_decomposition,
    get_inv_decomposition,
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
function posdef_correct(mat::AbstractMatrix; tol::Real = 1e12 * eps())
    mat = deepcopy(mat)
    if !issymmetric(mat)
        out = 0.5 * (mat + permutedims(mat, (2, 1))) #symmetrize
        if isposdef(out)
            # very often, small numerical errors cause asymmetry, so cheaper to add this branch
            return out
        end
    else
        out = mat
    end

    if !isposdef(out)
        nugget = abs(minimum(eigvals(out)))
        for i in 1:size(out, 1)
            out[i, i] += nugget + tol # add to diag
        end
    end
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

Stores a matrix along with a decomposition `T=Factor`, or pseudoinverse `T=PseInv`, and also computes the inverse of the Factored matrix (for several predictions this is actually the most computationally efficient action)


$(TYPEDFIELDS)
"""
struct Decomposition{T, M <: AbstractMatrix, MorF <: Union{AbstractMatrix, Factorization}}
    "The original matrix"
    full_matrix::M
    "The matrix decomposition, or pseudoinverse"
    decomposition::MorF
    "The matrix decomposition of the inverse, or pseudoinverse"
    inv_decomposition::M
end

function Decomposition(
    mat::M,
    method::S;
    nugget::R = 1e12 * eps(),
) where {M <: AbstractMatrix, S <: AbstractString, R <: Real}
    # TODOs
    # 1. Originally I used  f = getfield(LinearAlgebra, Symbol(method)) but this is slow for evaluation so defining svd and cholesky is all we have now. I could maybe do dispatch here to make this a bit more slick.
    # 2. I have tried using the in-place methods, but so far these have not made enough difference to be worthwhile, I think at some-point they would be, but the original matrix would be needed for matrix regularization. They are not the bottleneck in the end

    if method == "pinv"
        invmat = pinv(mat)
        return Decomposition{PseInv, M, M}(mat, invmat, invmat)
    elseif method == "svd"
        fmat = svd(mat)
        return Decomposition{Factor, typeof(mat), Base.return_types(svd, (typeof(mat),))[1]}(mat, fmat, inv(fmat))
    elseif method == "cholesky"
        if !isposdef(mat)
            #            @info "Random Feature system not positive definite. Performing cholesky factorization with a close positive definite matrix"
            mat = posdef_correct(mat, tol = nugget)
        end
        fmat = cholesky(mat)
        return Decomposition{Factor, typeof(mat), Base.return_types(cholesky, (typeof(mat),))[1]}(mat, fmat, inv(fmat))

    else
        throw(
            ArgumentError(
                "Only factorization methods \"pinv\", \"cholesky\" and \"svd\" implemented. got " * string(method),
            ),
        )

    end
end
"""
$(TYPEDSIGNATURES)

get `decomposition` field
"""
get_decomposition(d::Decomposition) = d.decomposition

"""
$(TYPEDSIGNATURES)

get `inv_decomposition` field
"""
get_inv_decomposition(d::Decomposition) = d.inv_decomposition

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
function linear_solve(
    d::Decomposition,
    rhs::A,
    ::Type{Factor};
    tullio_threading = true,
) where {A <: AbstractArray{<:AbstractFloat, 3}}
    # return get_decomposition(d) \ permutedims(rhs (3,1,2))
    # for prediction its far more worthwhile to store the inverse (in cases seen thus far)
    x = similar(rhs)#zeros(N, P, M)
    if !tullio_threading
        @tullio threads = 10^9 x[n, p, i] = get_inv_decomposition(d)[i, j] * rhs[n, p, j]
    else
        @tullio x[n, p, i] = get_inv_decomposition(d)[i, j] * rhs[n, p, j]
    end
    return x
end

function linear_solve(
    d::Decomposition,
    rhs::A,
    ::Type{PseInv};
    tullio_threading = true,
) where {A <: AbstractArray{<:AbstractFloat, 3}}
    # return get_decomposition(d) * rhs
    if !tullio_threading
        @tullio threads = 10^9 x[n, p, m] := get_decomposition(d)[m, i] * rhs[n, p, i]
    else
        @tullio x[n, p, m] := get_decomposition(d)[m, i] * rhs[n, p, i]
    end

    return x
end


function linear_solve(d::Decomposition, rhs::A, ::Type{Factor}; kwargs...) where {A <: AbstractVector{<:AbstractFloat}}
    # return get_decomposition(d) \ rhs
    return get_inv_decomposition(d) * rhs

end

function linear_solve(d::Decomposition, rhs::A, ::Type{PseInv}; kwargs...) where {A <: AbstractVector{<:AbstractFloat}}
    #get_decomposition(d) * rhs
    return get_decomposition(d) * rhs
end

linear_solve(d::Decomposition, rhs::A; tullio_threading = true) where {A <: AbstractArray} =
    linear_solve(d, rhs, get_parametric_type(d), tullio_threading = tullio_threading)

end
