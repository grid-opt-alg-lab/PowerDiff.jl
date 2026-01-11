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

    return VoltageSensitivityTopology(Matrix(sensitivities_g), Matrix(sensitivities_b))
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
