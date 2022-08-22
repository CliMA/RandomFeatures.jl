module Utilities

using LinearAlgebra

export
    batch_generator,
    Decomposition,
    get_decomposition_is_inverse,
    get_decomposition,
    get_full_matrix,
    linear_solve

"""
    function batch_generator(array::AbstractArray,batch_size::Int;dims::Int=1)

returns a vector of batched sub-array views of size `batch_size` along dimension `dims`.
NOTE: it creates views not copies. Modifying a batch will modify the original!

"""  
function batch_generator(
    array::AbstractArray,
    batch_size::Int;
    dims::Int=1
)
    if batch_size == 0 
        return [array]
    end
    
    n_batches = Int(ceil(size(array,dims)/batch_size))
    batch_idx = [
        i < n_batches ?
        collect((i-1)*batch_size+1:i*batch_size) :
        collect((i-1)*batch_size+1:size(array,dims))
        for i in 1:n_batches
    ]
    
    return [selectdim(array, dims, b) for b in batch_idx]
end
    

# Decomposition/linear solves for feature matrices

struct Decomposition
    full_matrix::AbstractMatrix
    decomposition::Union{AbstractMatrix,Factorization}
    decomposition_is_inverse::Bool
end

function Decomposition(mat::AbstractMatrix, method::AbstractString)
    if method == "pinv"
        decomposition = pinv(mat)
        decomposition_is_inverse = true
    else
        if !isdefined(LinearAlgebra,Symbol(method))
            throw(ArgumentError("factorization method "*string(method)*" not found in LinearAlgebra, please select one of the existing options"))
        else
            f = getfield(LinearAlgebra,Symbol(method))
            decomposition = f(mat)
            decomposition_is_inverse = false
        end

    end
    return Decomposition(mat, decomposition, decomposition_is_inverse)
end
get_decomposition_is_inverse(d::Decomposition) = d.decomposition_is_inverse
get_decomposition(d::Decomposition) = d.decomposition
get_full_matrix(d::Decomposition) = d.full_matrix

function linear_solve(d::Decomposition, rhs::AbstractVecOrMat)
    decomp = get_decomposition(d)
    if get_decomposition_is_inverse(d)
        #in this case the stored decomposition IS the (pseudo)inverse
        return decomp * rhs
    else
        #in this case the stored decomposition is just a factorization
        return decomp \ rhs
    end
end



end   
