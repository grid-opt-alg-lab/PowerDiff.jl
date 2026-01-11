###################################################################
### Structs for power flow equations:
### Vectorized, admittance-matrix-free, topology-agnostic power flow equations.

"""
Encodes a network topology and the corresponding admittance matrix.
Maps a sparse vectorized admittance to a corresponding network topology and admittance matrix.
The incidence matrix is a sparse matrix that encodes the connectivity of the network.
"""
mutable struct NetworkTopology
    A::SparseMatrixCSC{Int64,Int64} # incidence matrix
    g::AbstractVector{Float64} # edge-conductance vector of the network, including self-edges
    b::AbstractVector{Float64} # edge-susceptance vector of the network, including self-edges
    g_self::AbstractVector{Float64} # self-edge-conductance vector of the network (note this is appended to G)
    b_self::AbstractVector{Float64} # self-edge-susceptance vector of the network (note this is appended to B)
    has_self_edges::Bool # if true, self-edges are included in the incidence matrix
end

"""
Struct that encodes differentiable forms of the power flow equations.
"""
mutable struct PowerFlowEquations
    topology::NetworkTopology # network topology
    p::Function # function to compute real power injections
    q::Function # function to compute reactive power injections
    v_re::Function # function to compute the real part of voltage phasors
    v_im::Function # function to compute the imaginary part of voltage phasors
    vm::Function # function to compute the voltage magnitudes
    vm2::Function # function to compute the square of the voltage magnitudes
    branch_flow::Function # function to compute the branch flows
    p_flow::Function # function to compute the real part of the branch flows
    q_flow::Function # function to compute the imaginary part of the branch flows
end


"""
Given a PowerModels data dictionary, constructs differentiable power flow equations.
"""
function PowerFlowEquations(net::Dict{String,<:Any})
    # Construct the incidence matrix using the full_nodes/full_edges flags
    A = calc_incidence_matrix(net,full_nodes=true,full_edges=false)
    
    # Construct the vectorized weights.
    G,B = calc_vectorized_admittance_matrix(net)
    
    # Get the self edges
    Y_self = [sum(r) for r in eachrow(PM.calc_basic_admittance_matrix(net))]
    G_self = real.(Y_self)
    B_self = imag.(Y_self)

    topology = NetworkTopology(A,G,B,G_self,B_self,true)

    pf_eqns = PowerFlowEquations(
        topology,
        p(v_re,v_im,G,B),
        q(v_re,v_im,G,B),
        vm(v_re,v_im,G,B),
        vm2(v_re,v_im,G,B),
        branch_flow(v,G,B),
        p_flow(v,G,B),
        q_flow(v,G,B)
    )

    return pf_eqns

end