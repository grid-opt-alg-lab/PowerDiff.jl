using BlockArrays

"""
Generic construction of Laplacian matrix with vectorized weights and the full incidence matrix.
"""
function laplacian(G,B,n)
    # construct the incidence matrix
    A = full_incidence_matrix(n)
    # construct the Laplacian
    W = sparse(Diagonal(G .+ B .* im))
    return A'*W*A
end

"""
The full incidence matrix of an arbitrary n node network.
Note that every node has a self edge.
This assumes that there exists edges such that all nodes are connected to all other nodes.
"""
function full_incidence_matrix(n::Int;self_edges=true)
    dims = ((n^2+n)÷2 ,n)
    A = BlockedArray(spzeros(dims),vcat([dᵢ for dᵢ in n-1:-1:1],[n]),[n])
    for i in 1:n-1
        blocks(A)[i][:,i] .= 1
        blocks(A)[i][:,i+1:end] .= -I(n-i)
        #A[Block(i)][:,i] .= 1
        #A[Block(i)][:,i+1:end] .= -I(n-i)
    end
    # Add or remove the self edges
    if self_edges
        A[Block(n)] .= I(n)
    else
        A = A[1:n*(n-1)÷2,:]
    end
    return sparse(A)
end

"""
The full incidence matrix of a PowerModels network.
Note that every node has a self edge.
This assumes that there exists edges such that all nodes are connected to all other nodes.
"""
function full_incidence_matrix(net::Dict)
    num_bus = length(net["bus"])
    A_branch = calc_basic_incidence_matrix(net)
    return sparse_vcat(A_branch,I(num_bus))
end


"""
The full incidence matrix of a PowerModels network.
Note that every node has a self edge.
This uses the incidence matrix of the network to only consider edges that exist in the network.

Params:
    net: a PowerModels network
    full_nodes: if true, include the self edges -- append an identity matrix to the incidence matrix
    full_edges: if true, consider all possible edges -- use the full incidence matrix
"""
function calc_incidence_matrix(net::Dict;
    full_nodes::Bool=true,full_edges::Bool=false)

    num_bus = length(net["bus"])
    num_branch = length(net["branch"])
    A_branch = calc_basic_incidence_matrix(net)
    
    if full_nodes && !full_edges
        return sparse_vcat(A_branch,I(num_bus))
    elseif full_nodes && full_edges
        return full_incidence_matrix(num_bus)
    elseif !full_nodes && !full_edges
        return A_branch
    else
        error("The incidence matrix cannot be constructed with full_nodes=false and full_edges=true.")
    end
end

function make_vectorized_laplacian_indices(Y::AbstractMatrix)
    LR = tril(ones(size(Y)),-1)
    idxs = CartesianIndex[]
    for ix in CartesianIndices(LR)
        if LR[ix] == 1
            push!(idxs,ix)
        end
    end
    return idxs
end


"""
Construct vectorized Laplacian weights from the admittance matrix. Assume that every node has a self edge.
"""
function vectorize_laplacian_weights(Y::AbstractMatrix{ComplexF64})

    @assert Y == transpose(Y) "The admittance matrix must be symmetric."
    
    # Make the vectorized weights
    G_full = real.(Y)
    B_full = imag.(Y)
    
    # Get the strictly off-diagonal entries of the lower triangular part of the Laplacian, excluding the diagonal.
    # Vectorize the weights.
    G_edges = -1 .* G_full[tril(ones(size(G_full)),-1) .== 1]
    B_edges = -1 .* B_full[tril(ones(size(B_full)),-1) .== 1]

    # Get the self edges
    Y_self = [sum(r) for r in eachrow(Y)]
    G_self = real.(Y_self)
    B_self = imag.(Y_self)

    # Construct the vectorized weights.
    G = vcat(G_edges,G_self)
    B = vcat(B_edges,B_self)

    @assert norm(laplacian(G,B,size(Y,1)) -Y) < 1e-6 "The Laplacian matrix is not close to the admittance matrix. The norm of the difference is $(norm(laplacian(G,B,size(Y,1)) -Y))."

    return G,B
end


# vectorize_laplacian_weights(net::Dict{String,Any}) = vectorize_laplacian_weights(calc_basic_admittance_matrix(net))
"""
Construct vectorized laplacian weights from a PowerModels data dictionary.
"""
function vectorize_laplacian_weights(
    net::Dict;
    full_nodes::Bool=true,
    full_edges::Bool=false
)

    num_bus = length(net["bus"])
    num_branch = length(net["branch"])
    Y_mat = calc_basic_admittance_matrix(net)

    # Construct the incidence matrix using the full_nodes/full_edges flags
    A = calc_incidence_matrix(net,full_nodes=full_nodes,full_edges=full_edges)

    
    # Construct the vectorized weights.
    if !full_edges # standard case - no self edges, no full incidence matrix, just use the edges as standard.

        G = full_nodes ? spzeros(num_branch+num_bus) : spzeros(num_branch)
        B = full_nodes ? spzeros(num_branch+num_bus) : spzeros(num_branch)
        for br_label = 1:num_branch
            br = net["branch"]["$(br_label)"]
            ix_br,f_bus,t_bus = br["index"],br["f_bus"],br["t_bus"]
            G[ix_br] = -1*real(Y_mat[f_bus,t_bus])
            B[ix_br] = -1*imag(Y_mat[f_bus,t_bus])
        end

        if full_nodes
            # Get the self edges
            Y_self = [sum(r) for r in eachrow(Y_mat)]
            for i = 1:num_bus
                G[num_branch+i] = real(Y_self[i])
                B[num_branch+i] = imag(Y_self[i]) 
            end
        end
        
    elseif full_edges
        # Get the strictly off-diagonal entries of the lower triangular part of the Laplacian (including zeos).
        Y_edges = -1 .* Y_mat[tril(ones(size(Y_mat)),-1) .== 1]
        # Get the self edges
        Y_self = [sum(r) for r in eachrow(Y_mat)]
        # Construct the vectorized weights.
        Y = full_nodes ? vcat(Y_edges,Y_self) : Y_edges
        G = real.(Y)
        B = imag.(Y)

    end

    Y_recon = A'*Diagonal(G + B*im)*A
    @assert norm(Y_recon - Y_mat) < 1e-6 "The admittance matrix is not close to the vectorized admittance matrix. The norm of the difference is $(norm(Y_recon - Y_mat))."    


    return G,B
end



calc_branch_current(G::AbstractVector{ComplexF64},B::AbstractVector{ComplexF64},x::AbstractVector{ComplexF64}) = spdiagm(G + B.* im)*full_incidence_matrix(length(x))*x

"""
Given a basic network dict, a vectorized  admittance matrix and a bus voltage solution, compute the branch current.
"""
function calc_branch_current(net::Dict{String,<:Any},Y::VectorizedAdmittanceMatrix,x::AbstractVector)
    G,B = Y.G,Y.B
    vecY_idxs = make_vectorized_laplacian_indices(Y.matrix)
    num_branch = length(net["branch"]) 
    A = calc_incidence_matrix(net,full_nodes=true,full_edges=true) # The full incidence matrix, wth all possible edges
    ℓ_full = Diagonal(G + B*im)*A*x # the currents through all possible branches. Includes self edges
    ℓ_branch = zeros(ComplexF64,num_branch) # the branch currents, with every index non-zero. Only includes branches that actually exist
    for (_,br) in net["branch"]
        br_idx,f_bus,t_bus = br["index"],br["f_bus"],br["t_bus"]
        # Find the vecY cartesian index and corresponding linear index for this branch
        ij_full_ix = findall(ix -> ix[1] == f_bus && ix[2] == t_bus,vecY_idxs)
        ji_full_ix = findall(ix -> ix[1] == t_bus && ix[2] == f_bus,vecY_idxs)
        ℓ_branch[br_idx] = !isempty(ij_full_ix) ? -ℓ_full[ij_full_ix[1]] : ℓ_full[ji_full_ix[1]] # Note that the minus signs are swapped due to the admittance weights being the negative of the Y entries
    end
    return ℓ_branch
end

"""
Given a PowerModels network dict, compute the branch current flows. Assumes a solution to the power flow problem has been computed.
"""
function calc_branch_current(net::Dict)
    num_branch = length(net["branch"])
    G,B = vectorize_laplacian_weights(net,full_nodes=true,full_edges=false)
    A = calc_incidence_matrix(net,full_nodes=true,full_edges=false)
    x = calc_basic_bus_voltage(net)
    L = spdiagm(A*x)*(G + B*im) # current flows, includes self edges
    return L[1:num_branch] # remove self edges
end

