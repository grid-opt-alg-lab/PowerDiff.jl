# =============================================================================
# Power Flow Equations (Vectorized, Topology-Agnostic)
# =============================================================================
#
# Provides differentiable power flow equations using vectorized admittance
# representation. Equations are parameterized by G (conductance) and B (susceptance)
# vectors rather than a full Y matrix, enabling efficient implicit differentiation.

# =============================================================================
# Network Topology Struct
# =============================================================================

"""
    NetworkTopology

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
    NetworkTopology(net::Dict; full_nodes=true, full_edges=false)

Construct NetworkTopology from a PowerModels basic network.

Note: For networks with parallel branches, we compute individual branch admittances
from the branch impedance data rather than from the aggregated Y matrix.
"""
function NetworkTopology(net::Dict{String,<:Any}; full_nodes::Bool=true, full_edges::Bool=false)
    @assert haskey(net, "basic_network") && net["basic_network"] "Network must be a basic network"

    A = calc_incidence_matrix(net; full_nodes=full_nodes, full_edges=full_edges)
    num_branch = length(net["branch"])
    num_bus = length(net["bus"])

    # Compute individual branch admittances from branch data (handles parallel branches correctly)
    G = full_nodes ? zeros(num_branch + num_bus) : zeros(num_branch)
    B = full_nodes ? zeros(num_branch + num_bus) : zeros(num_branch)

    for (_, br) in net["branch"]
        ix = br["index"]
        r = br["br_r"]
        x = br["br_x"]

        # Branch admittance y = 1/(r + jx) = (r - jx) / (r^2 + x^2)
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

    # Self-edges (shunt admittances + line charging)
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

# =============================================================================
# Power Flow Equations - Complex Voltage Form
# =============================================================================

"""
    p(v, G, B)

Real power injection at each bus given complex voltage phasors.

Uses the vectorized admittance representation:
    P = Re(diag(v̄) * Y * v) where Y = A' * diag(G + jB) * A
"""
p(v, G, B) = real.(Diagonal(conj.(v)) * laplacian(G, B, length(v)) * v)

"""
    q(v, G, B)

Reactive power injection at each bus given complex voltage phasors.
"""
q(v, G, B) = imag.(Diagonal(conj.(v)) * laplacian(G, B, length(v)) * v)

"""
    vm(v, G, B)

Voltage magnitudes (G, B are unused but kept for interface consistency).
"""
vm(v, G, B) = abs.(v)

# =============================================================================
# Power Flow Equations - Rectangular Voltage Form
# =============================================================================

"""
    pf_eqns(v_re, v_im, G, B)

Complex power injection S = P + jQ in rectangular coordinates.
"""
pf_eqns(v_re, v_im, G, B) = Diagonal(v_re .- im .* v_im) * (laplacian(G, B, length(v_re)) * (v_re .+ im .* v_im))

"""
    p(v_re, v_im, G, B)

Real power injection in rectangular coordinates.
"""
p(v_re, v_im, G, B) = real.(pf_eqns(v_re, v_im, G, B))

"""
    q(v_re, v_im, G, B)

Reactive power injection in rectangular coordinates.
"""
q(v_re, v_im, G, B) = imag.(pf_eqns(v_re, v_im, G, B))

"""
    vm(v_re, v_im, G, B)

Voltage magnitudes from rectangular coordinates.
"""
vm(v_re, v_im, G, B) = hypot.(v_re, v_im)

"""
    vm2(v_re, v_im, G, B)

Squared voltage magnitudes from rectangular coordinates.
"""
vm2(v_re, v_im, G, B) = v_re.^2 .+ v_im.^2

# =============================================================================
# Power Flow Equations - Polar Form
# =============================================================================

"""
    p_polar(vm, δ, G, B)

Real power injection in polar coordinates (vm = voltage magnitude, δ = phase angle).
"""
p_polar(vm, δ, G, B) = p(vm .* cis.(δ), G, B)

"""
    q_polar(vm, δ, G, B)

Reactive power injection in polar coordinates.
"""
q_polar(vm, δ, G, B) = q(vm .* cis.(δ), G, B)

# =============================================================================
# Branch Flow Equations
# =============================================================================

"""
    branch_flow(v, G, B)

Complex branch current flows: I_branch = diag(A*v) * (G + jB)

Note: This uses the full incidence matrix and returns currents for all edges.
"""
branch_flow(v, G, B) = Diagonal(full_incidence_matrix(length(v)) * v) * (G .+ B .* im)

"""
    p_flow(v, G, B)

Real part of branch power flows.
"""
p_flow(v, G, B) = real.(branch_flow(v, G, B))

"""
    q_flow(v, G, B)

Reactive part of branch power flows.
"""
q_flow(v, G, B) = imag.(branch_flow(v, G, B))

# =============================================================================
# Power Flow Jacobians - Complex Voltage Form
# =============================================================================

"""Jacobian ∂P/∂G (real power w.r.t. conductance)"""
∂p∂g(v, G, B) = ForwardDiff.jacobian(G -> p(v, G, B), G)

"""Jacobian ∂P/∂B (real power w.r.t. susceptance)"""
∂p∂b(v, G, B) = ForwardDiff.jacobian(B -> p(v, G, B), B)

"""Jacobian ∂Q/∂G (reactive power w.r.t. conductance)"""
∂q∂g(v, G, B) = ForwardDiff.jacobian(G -> q(v, G, B), G)

"""Jacobian ∂Q/∂B (reactive power w.r.t. susceptance)"""
∂q∂b(v, G, B) = ForwardDiff.jacobian(B -> q(v, G, B), B)

"""Jacobian ∂|V|/∂G (voltage magnitude w.r.t. conductance) - always zero"""
∂vm∂g(v, G, B) = ForwardDiff.jacobian(G -> vm(v, G, B), G)

"""Jacobian ∂|V|/∂B (voltage magnitude w.r.t. susceptance) - always zero"""
∂vm∂b(v, G, B) = ForwardDiff.jacobian(B -> vm(v, G, B), B)

# =============================================================================
# Power Flow Jacobians - Polar Form
# =============================================================================

"""Jacobian ∂P/∂|V| in polar coordinates"""
∂p∂vm(vm, δ, G, B) = ForwardDiff.jacobian(vm -> p(vm .* cis.(δ), G, B), vm)

"""Jacobian ∂Q/∂|V| in polar coordinates"""
∂q∂vm(vm, δ, G, B) = ForwardDiff.jacobian(vm -> q(vm .* cis.(δ), G, B), vm)

"""Jacobian ∂P/∂δ in polar coordinates"""
∂p∂δ(vm, δ, G, B) = ForwardDiff.jacobian(δ -> p(vm .* cis.(δ), G, B), δ)

"""Jacobian ∂Q/∂δ in polar coordinates"""
∂q∂δ(vm, δ, G, B) = ForwardDiff.jacobian(δ -> q(vm .* cis.(δ), G, B), δ)

# =============================================================================
# Power Flow Jacobians - Rectangular Form
# =============================================================================

"""Jacobian ∂P/∂G in rectangular coordinates"""
∂p∂g(v_re, v_im, G, B) = ForwardDiff.jacobian(G -> p(v_re, v_im, G, B), G)

"""Jacobian ∂P/∂B in rectangular coordinates"""
∂p∂b(v_re, v_im, G, B) = ForwardDiff.jacobian(B -> p(v_re, v_im, G, B), B)

"""Jacobian ∂Q/∂G in rectangular coordinates"""
∂q∂g(v_re, v_im, G, B) = ForwardDiff.jacobian(G -> q(v_re, v_im, G, B), G)

"""Jacobian ∂Q/∂B in rectangular coordinates"""
∂q∂b(v_re, v_im, G, B) = ForwardDiff.jacobian(B -> q(v_re, v_im, G, B), B)

"""Jacobian ∂|V|/∂G in rectangular coordinates"""
∂vm∂g(v_re, v_im, G, B) = ForwardDiff.jacobian(G -> vm(v_re, v_im, G, B), G)

"""Jacobian ∂|V|/∂B in rectangular coordinates"""
∂vm∂b(v_re, v_im, G, B) = ForwardDiff.jacobian(B -> vm(v_re, v_im, G, B), B)

"""Jacobian ∂P/∂v_re (real power w.r.t. real voltage)"""
∂p∂v_re(v_re, v_im, G, B) = ForwardDiff.jacobian(v_re -> p(v_re, v_im, G, B), v_re)

"""Jacobian ∂P/∂v_im (real power w.r.t. imaginary voltage)"""
∂p∂v_im(v_re, v_im, G, B) = ForwardDiff.jacobian(v_im -> p(v_re, v_im, G, B), v_im)

"""Jacobian ∂Q/∂v_re (reactive power w.r.t. real voltage)"""
∂q∂v_re(v_re, v_im, G, B) = ForwardDiff.jacobian(v_re -> q(v_re, v_im, G, B), v_re)

"""Jacobian ∂Q/∂v_im (reactive power w.r.t. imaginary voltage)"""
∂q∂v_im(v_re, v_im, G, B) = ForwardDiff.jacobian(v_im -> q(v_re, v_im, G, B), v_im)

# =============================================================================
# PowerFlowEquations Struct (Closures with Fixed Topology)
# =============================================================================

"""
    PowerFlowEquations

Differentiable power flow equations with fixed network topology.

Stores closures that compute power flow quantities using the network's actual
incidence matrix (not the full all-to-all incidence matrix). This enables
differentiation w.r.t. actual branch parameters.

# Fields
- `topology`: NetworkTopology struct
- `p`: (v_re, v_im) -> real power injection
- `q`: (v_re, v_im) -> reactive power injection
- `vm`: (v_re, v_im) -> voltage magnitudes
- `vm2`: (v_re, v_im) -> squared voltage magnitudes
- `branch_flow`: (v) -> complex branch currents
- `p_flow`: (v) -> real branch power flows
- `q_flow`: (v) -> reactive branch power flows

# Example
```julia
pf = PowerFlowEquations(net)
v_re = real.(v)
v_im = imag.(v)
P = pf.p(v_re, v_im)
Q = pf.q(v_re, v_im)
```
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
    PowerFlowEquations(net::Dict)

Construct PowerFlowEquations from a PowerModels basic network.

The returned struct contains closures that compute power flow using the
network's actual incidence matrix (handles arbitrary topologies).
"""
function PowerFlowEquations(net::Dict{String,<:Any})
    topology = NetworkTopology(net)
    return PowerFlowEquations(topology)
end

"""
    PowerFlowEquations(topology::NetworkTopology)

Construct PowerFlowEquations from an existing NetworkTopology.

Uses the stored incidence matrix A to compute power flow as:
    S = diag(v̄) * Y * v where Y = A' * diag(G + jB) * A
"""
function PowerFlowEquations(topology::NetworkTopology)
    A = topology.A
    G = topology.G
    B = topology.B

    # Build the Y matrix from A' * W * A
    function compute_Y()
        W = sparse(Diagonal(G .+ B .* im))
        return transpose(A) * W * A
    end

    # Power injection: S = diag(v̄) * Y * v
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

    # Branch flow: I_branch = diag(y) * A * v (current through each edge)
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
