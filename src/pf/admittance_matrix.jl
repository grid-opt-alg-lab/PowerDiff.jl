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

### Vectorized admittance matrix structure
struct VectorizedAdmittanceMatrix
    matrix::AbstractMatrix{ComplexF64} # admittance matrix
    G::AbstractVector{Float64} # edge-conductance vector of the network, including self-edges
    B::AbstractVector{Float64} # edge-susceptance vector of the network, including self-edges
    G_self::AbstractVector{Float64} # self-edge-conductance vector of the network (note this is appended to G)
    B_self::AbstractVector{Float64} # self-edge-susceptance vector of the network (note this is appended to B)
end
    
"""
Given an admittance matrix, construct a VectorizedAdmittanceMatrix.
"""
function VectorizedAdmittanceMatrix(Y::AbstractMatrix{ComplexF64})
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

    return VectorizedAdmittanceMatrix(Y,G,B,G_self,B_self)
end


"""
Given: 
    G: edge-conductance vector of the network, including self-edges
    B: edge-susceptance vector of the network, including self-edges
    n_bus: number of buses in the network
 construct a VectorizedAdmittanceMatrix.
"""
function VectorizedAdmittanceMatrix(G::AbstractVector{Float64},B::AbstractVector{Float64},n_bus::Int)
    Y = laplacian(G,B,n_bus)
    Y_self = [sum(r) for r in eachrow(Y)]
    G_self = real.(Y_self)
    B_self = imag.(Y_self)
    return VectorizedAdmittanceMatrix(Y,G,B,G_self,B_self)
end