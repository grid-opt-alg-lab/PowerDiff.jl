# =============================================================================
# DEPRECATED: NetworkTopology and PowerFlowEquations
# =============================================================================
#
# These types are deprecated in favor of ACNetwork which provides the same
# functionality with a cleaner Julian design (separate state and method).
#
# Use ACNetwork instead:
#   net = ACNetwork(pm_data)
#   P = p(net, v)
#   Q = q(net, v)
#
# See src/types/ac_network.jl for the new implementation.

"""
    NetworkTopology (DEPRECATED)

Use `ACNetwork` instead. This type is kept for backwards compatibility.

Encodes a network topology and the corresponding vectorized admittance representation.

# Fields
- `A`: Incidence matrix (m × n) or extended incidence matrix including self-edges
- `G`: Edge conductance vector (includes self-edges if `has_self_edges=true`)
- `B`: Edge susceptance vector (includes self-edges if `has_self_edges=true`)
- `G_self`: Self-edge conductance vector (diagonal of Y_bus)
- `B_self`: Self-edge susceptance vector (diagonal of Y_bus)
- `has_self_edges`: If true, self-edges are included in G, B and A
"""
mutable struct NetworkTopology
    A::SparseMatrixCSC{Float64,Int64}
    G::Vector{Float64}
    B::Vector{Float64}
    G_self::Vector{Float64}
    B_self::Vector{Float64}
    has_self_edges::Bool
end

"""
    NetworkTopology(net::Dict; full_nodes=true, full_edges=false) (DEPRECATED)

Use `ACNetwork(net)` instead.
"""
function NetworkTopology(net::Dict{String,<:Any}; full_nodes::Bool=true, full_edges::Bool=false)
    @assert haskey(net, "basic_network") && net["basic_network"] "Network must be a basic network"

    A = calc_incidence_matrix(net; full_nodes=full_nodes, full_edges=full_edges)
    num_branch = length(net["branch"])
    num_bus = length(net["bus"])

    G = full_nodes ? zeros(num_branch + num_bus) : zeros(num_branch)
    B = full_nodes ? zeros(num_branch + num_bus) : zeros(num_branch)

    for (_, br) in net["branch"]
        ix = br["index"]
        r = br["br_r"]
        x = br["br_x"]

        z2 = r^2 + x^2
        if z2 > 1e-10
            g_branch = r / z2
            b_branch = -x / z2
        else
            g_branch = 0.0
            b_branch = 0.0
        end

        G[ix] = g_branch
        B[ix] = b_branch
    end

    Y_mat = PM.calc_basic_admittance_matrix(net)
    Y_self = [sum(r) for r in eachrow(Y_mat)]
    G_self = real.(Y_self)
    B_self = imag.(Y_self)

    if full_nodes
        for i = 1:num_bus
            G[num_branch + i] = G_self[i]
            B[num_branch + i] = B_self[i]
        end
    end

    return NetworkTopology(Float64.(A), Vector(G), Vector(B), G_self, B_self, full_nodes)
end

"""
    PowerFlowEquations (DEPRECATED)

Use functions on `ACNetwork` instead. This type uses function fields which is
not idiomatic Julia.

Instead of:
    pf = PowerFlowEquations(net)
    P = pf.p(v_re, v_im)

Use:
    ac_net = ACNetwork(net)
    P = p(ac_net, v)
"""
struct PowerFlowEquations
    topology::NetworkTopology
    p::Function
    q::Function
    vm::Function
    vm2::Function
    branch_flow::Function
    p_flow::Function
    q_flow::Function
end

"""
    PowerFlowEquations(net::Dict) (DEPRECATED)

Use `ACNetwork(net)` instead.
"""
function PowerFlowEquations(net::Dict{String,<:Any})
    topology = NetworkTopology(net)
    return PowerFlowEquations(topology)
end

"""
    PowerFlowEquations(topology::NetworkTopology) (DEPRECATED)
"""
function PowerFlowEquations(topology::NetworkTopology)
    A = topology.A
    G = topology.G
    B = topology.B

    function compute_Y()
        W = sparse(Diagonal(G .+ B .* im))
        return transpose(A) * W * A
    end

    function p_func(v_re, v_im)
        v = v_re .+ im .* v_im
        Y = compute_Y()
        S = Diagonal(conj.(v)) * Y * v
        return real.(S)
    end

    function q_func(v_re, v_im)
        v = v_re .+ im .* v_im
        Y = compute_Y()
        S = Diagonal(conj.(v)) * Y * v
        return imag.(S)
    end

    function vm_func(v_re, v_im)
        return hypot.(v_re, v_im)
    end

    function vm2_func(v_re, v_im)
        return v_re.^2 .+ v_im.^2
    end

    function branch_flow_func(v)
        W = sparse(Diagonal(G .+ B .* im))
        return W * A * v
    end

    function p_flow_func(v)
        return real.(branch_flow_func(v))
    end

    function q_flow_func(v)
        return imag.(branch_flow_func(v))
    end

    return PowerFlowEquations(
        topology,
        p_func,
        q_func,
        vm_func,
        vm2_func,
        branch_flow_func,
        p_flow_func,
        q_flow_func
    )
end
