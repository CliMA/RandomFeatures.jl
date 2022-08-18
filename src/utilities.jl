module utilities

export
    batch_generator

function batch_generator(
    input::AbstractArray,
    batch_size::Int;
    dims::Int=1
)
    n_batches = Int(ceil(size(input,dims)/batch_size))
    batch_idx = [
        i < n_batches ?
        collect((i-1)*batch_size+1:i*batch_size) :
        collect((i-1)*batch_size+1:size(input,dims))
        for i in 1:n_batches
    ]
    
    return [selectdim(input, dims, b) for b in batch_idx]
end
    
end   
