"""
    voltage_topology_sensitivities(net; voltages=nothing, full_nodes=true,
        full_edges=false, check_solution=true)

Compute the linearized sensitivity of bus-voltage magnitudes with respect to
perturbations in the network topology parameters of a PowerModels-style network.

The function assumes that `net` has been converted with `make_basic_network`
and that a steady-state power flow solution is available. If `voltages` is not
provided, the solved bus voltages are extracted via `PowerModels.calc_basic_bus_voltage`.

Arguments
---------
- `net`: PowerModels network dictionary with `"basic_network" == true`.
- `voltages`: Optional complex vector of bus voltages (defaults to the solution
  stored in `net`).
- `full_nodes`: When `true`, sensitivities include shunt/self-edge parameters in
  the vectorized admittance (default).
- `full_edges`: When `true`, sensitivities are reported for all possible
  off-diagonal edges produced by `vectorize_laplacian_weights`.
- `check_solution`: Validate that the provided voltages satisfy the base power
  flow equations up to a 1e-6 mismatch.

Returns
-------
`VoltageSensitivityTopology` whose fields `p` and `q` store the Jacobians
∂|V|/∂G and ∂|V|/∂B respectively, matching the ordering produced by
`vectorize_laplacian_weights(net; full_nodes, full_edges)`.
"""
function voltage_topology_sensitivities(
    net::Dict{String,<:Any};
    voltages::Union{Nothing,AbstractVector{<:Complex}}=nothing,
    full_nodes::Bool=true,
    full_edges::Bool=false,
    check_solution::Bool=true,
)
    net["basic_network"] == true ||
        throw(ArgumentError("network must be built with `make_basic_network`"))

    base_v = isnothing(voltages) ? PM.calc_basic_bus_voltage(net) : voltages
    n_bus = length(net["bus"])
    length(base_v) == n_bus ||
        throw(ArgumentError("voltages length $(length(base_v)) does not match "
            * "number of buses $n_bus"))

    v_re = Float64.(real.(base_v))
    v_im = Float64.(imag.(base_v))
    vm = hypot.(v_re, v_im)

    G, B = vectorize_laplacian_weights(net; full_nodes=full_nodes, full_edges=full_edges)

    check_solution && _assert_pf_solution(net, v_re, v_im, G, B)

    J_state = _voltage_state_jacobian(v_re, v_im, G, B)
    J_g = _topology_jacobian(v_re, v_im, G, B, :g)
    J_b = _topology_jacobian(v_re, v_im, G, B, :b)

    lu_state = lu(J_state)

    Δstate_g = -(lu_state \ J_g)
    Δstate_b = -(lu_state \ J_b)

    sensitivities_g = _vm_projection(vm, v_re, v_im, Δstate_g)
    sensitivities_b = _vm_projection(vm, v_re, v_im, Δstate_b)

    return VoltageTopologySensitivity(Matrix(sensitivities_g), Matrix(sensitivities_b))
end

function _assert_pf_solution(
    net::Dict{String,<:Any},
    v_re::Vector{Float64},
    v_im::Vector{Float64},
    G::AbstractVector{<:Real},
    B::AbstractVector{<:Real};
    atol::Float64=1e-6,
)
    injections = PM.calc_basic_bus_injection(net)
    active_target = real.(injections)
    reactive_target = imag.(injections)

    active_eval = var"p"(v_re, v_im, G, B)
    reactive_eval = var"q"(v_re, v_im, G, B)

    mismatch_p = maximum(abs.(active_eval .- active_target))
    mismatch_q = maximum(abs.(reactive_eval .- reactive_target))

    (mismatch_p ≤ atol && mismatch_q ≤ atol) || throw(ArgumentError(
        "supplied voltages do not satisfy the stored power flow solution "
        * "(active mismatch=$mismatch_p, reactive mismatch=$mismatch_q)"))

    return nothing
end

function _voltage_state_jacobian(
    v_re::Vector{Float64},
    v_im::Vector{Float64},
    G::AbstractVector{<:Real},
    B::AbstractVector{<:Real},
)
    J_p_vre = var"∂p∂v_re"(v_re, v_im, G, B)
    J_p_vim = var"∂p∂v_im"(v_re, v_im, G, B)
    J_q_vre = var"∂q∂v_re"(v_re, v_im, G, B)
    J_q_vim = var"∂q∂v_im"(v_re, v_im, G, B)
    return [J_p_vre J_p_vim; J_q_vre J_q_vim]
end

function _topology_jacobian(
    v_re::Vector{Float64},
    v_im::Vector{Float64},
    G::AbstractVector{<:Real},
    B::AbstractVector{<:Real},
    mode::Symbol,
)
    if mode === :g
        J_p = var"∂p∂g"(v_re, v_im, G, B)
        J_q = var"∂q∂g"(v_re, v_im, G, B)
    elseif mode === :b
        J_p = var"∂p∂b"(v_re, v_im, G, B)
        J_q = var"∂q∂b"(v_re, v_im, G, B)
    else
        throw(ArgumentError("unsupported topology sensitivity mode $mode"))
    end
    return [J_p; J_q]
end

function _vm_projection(
    vm::Vector{Float64},
    v_re::Vector{Float64},
    v_im::Vector{Float64},
    Δstate::AbstractMatrix{<:Real},
)
    n = length(vm)
    vm_safe = max.(vm, eps(Float64))
    scaling_re = Diagonal(v_re ./ vm_safe)
    scaling_im = Diagonal(v_im ./ vm_safe)
    Δv_re = Δstate[1:n, :]
    Δv_im = Δstate[n+1:end, :]
    return scaling_re * Δv_re + scaling_im * Δv_im
end

# =============================================================================
# DC Power Flow Switching Sensitivity
# =============================================================================

"""
    calc_sensitivity_switching(state::DCPowerFlowState) → DCPFSwitchingSens

Compute switching sensitivity for DC power flow (not OPF).

For DC power flow θ = L(z)⁺ p, the sensitivity of angles w.r.t. switching is:

    ∂θ/∂zₑ = -L⁺ · (∂L/∂zₑ) · θ

where ∂L/∂zₑ = -bₑ · (aₑ · aₑ') is the rank-1 outer product of incidence column.

This uses the formula from matrix perturbation theory (RandomizedSwitching pattern).

# Arguments
- `state`: DCPowerFlowState containing the solved power flow

# Returns
`DCPFSwitchingSens` with:
- `dva_dz`: Jacobian ∂va/∂z (n × m) - voltage angles w.r.t. switching
- `df_dz`: Jacobian ∂f/∂z (m × m) - flows w.r.t. switching
"""
function calc_sensitivity_switching(state::DCPowerFlowState)
    net = state.net
    n, m = net.n, net.m
    ref = net.ref_bus

    # Use cached pseudoinverse from state (O(1) instead of O(n³))
    L_pinv = state.L_pinv

    # Balance p at slack bus as in DCPowerFlowState constructor
    p_balanced = copy(state.p)
    p_balanced[ref] = -sum(state.p) + state.p[ref]

    # Compute raw θ (before centering)
    θ_raw = L_pinv * p_balanced

    # Preallocate
    dva_dz = zeros(n, m)

    # For each edge e, compute ∂va/∂zₑ
    for e in 1:m
        # Get incidence column for edge e: a_e = A[e, :]
        # Note: A is m × n, so we get the e-th row
        aₑ = Vector(net.A[e, :])

        # ∂L/∂zₑ = -bₑ · (aₑ · aₑ')
        # This is a rank-1 matrix
        ∂L_∂zₑ = -net.b[e] * (aₑ * aₑ')

        # ∂va_raw/∂zₑ = -L⁺ · ∂L/∂zₑ · va_raw
        dva_raw_dzₑ = -L_pinv * ∂L_∂zₑ * θ_raw

        # Account for centering: va = va_raw - va_raw[ref]
        # So ∂va/∂zₑ = ∂va_raw/∂zₑ - (∂va_raw/∂zₑ)[ref] · 1
        dva_dz[:, e] = dva_raw_dzₑ .- dva_raw_dzₑ[ref]
    end

    # Flow sensitivity: f = W · A · va where W = Diag(-b ⊙ z)
    # fₑ = -bₑ · zₑ · (A[e,:] · va)
    #
    # ∂fₑ/∂zₑ' has two components:
    # 1. Direct effect (if e' = e): ∂fₑ/∂zₑ = -bₑ · (A[e,:] · va)
    # 2. Indirect effect via va: ∂fₑ/∂zₑ' = -bₑ · zₑ · (A[e,:] · ∂va/∂zₑ')
    df_dz = zeros(m, m)

    W = Diagonal(-net.b .* net.z)
    for e_prime in 1:m
        # Indirect effect: all edges feel the change in va
        df_dz[:, e_prime] = W * net.A * dva_dz[:, e_prime]

        # Direct effect: only edge e_prime
        df_dz[e_prime, e_prime] += -net.b[e_prime] * dot(net.A[e_prime, :], state.θ)
    end

    return DCPFSwitchingSens(dva_dz, df_dz)
end

"""
    calc_sensitivity_demand(state::DCPowerFlowState) → DCPFDemandSens

Compute demand sensitivity for DC power flow (not OPF).

For DC power flow va = L(z)⁺ p, the sensitivity of angles w.r.t. demand is:

    ∂va/∂d = -L⁺

since p = g - d and ∂p/∂d = -I.

# Arguments
- `state`: DCPowerFlowState containing the solved power flow

# Returns
`DCPFDemandSens` with:
- `dva_dd`: Jacobian ∂va/∂d (n × n) - voltage angles w.r.t. demand
- `df_dd`: Jacobian ∂f/∂d (m × n) - flows w.r.t. demand
"""
function calc_sensitivity_demand(state::DCPowerFlowState)
    net = state.net

    # Use cached pseudoinverse from state (O(1) instead of O(n³))
    # ∂va/∂d = -L⁺ (since ∂p/∂d = -I and va = L⁺ p)
    dva_dd = -state.L_pinv

    # ∂f/∂d = W · A · ∂va/∂d
    W = Diagonal(-net.b .* net.z)
    df_dd = W * net.A * dva_dd

    return DCPFDemandSens(dva_dd, df_dd)
end
