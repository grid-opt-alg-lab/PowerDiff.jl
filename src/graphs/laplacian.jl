# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generic construction of Laplacian matrix with vectorized weights and the full incidence matrix.
"""
function laplacian(G,B,n)
    # construct the incidence matrix
    A = complete_incidence_matrix(n)
    # construct the Laplacian
    W = sparse(Diagonal(G .+ B .* im))
    return A'*W*A
end

"""
The full incidence matrix of an arbitrary n node network.
Note that every node has a self edge.
This assumes that there exists edges such that all nodes are connected to all other nodes.
"""
function complete_incidence_matrix(n::Int;self_edges=true)
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
    calc_incidence_matrix(net::Dict; full_nodes=true, full_edges=false)

The full incidence matrix of a PowerModels network.
Note that every node has a self edge.
This uses the incidence matrix of the network to only consider edges that exist in the network.

!!! note "Requires basic network"
    This function calls `PM.calc_basic_incidence_matrix` internally,
    which requires `net` to be converted with `make_basic_network` first.

Params:
    net: a PowerModels network (must be basic)
    full_nodes: if true, include the self edges -- append an identity matrix to the incidence matrix
    full_edges: if true, consider all possible edges -- use the full incidence matrix
"""
function calc_incidence_matrix(net::Dict;
    full_nodes::Bool=true,full_edges::Bool=false)

    num_bus = length(net["bus"])
    A_branch = PM.calc_basic_incidence_matrix(net)
    
    if full_nodes && !full_edges
        return sparse_vcat(A_branch,I(num_bus))
    elseif full_nodes && full_edges
        return complete_incidence_matrix(num_bus)
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

    Y == transpose(Y) || throw(ArgumentError("The admittance matrix must be symmetric."))
    
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

    residual = norm(laplacian(G, B, size(Y, 1)) - Y)
    residual < 1e-6 || throw(ArgumentError("Laplacian reconstruction error: $residual"))

    return G,B
end


"""
    vectorize_laplacian_weights(net::Dict; full_nodes=true, full_edges=false)

Construct vectorized laplacian weights from a PowerModels data dictionary.

!!! note "Requires basic network"
    This function calls `PM.calc_basic_admittance_matrix` internally,
    which requires `net` to be converted with `make_basic_network` first.
"""
function vectorize_laplacian_weights(
    net::Dict;
    full_nodes::Bool=true,
    full_edges::Bool=false
)

    num_bus = length(net["bus"])
    num_branch = length(net["branch"])
    Y_mat = PM.calc_basic_admittance_matrix(net)

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
    residual = norm(Y_recon - Y_mat)
    residual < 1e-6 || throw(ArgumentError("Vectorized admittance reconstruction error: $residual"))


    return G,B
end


